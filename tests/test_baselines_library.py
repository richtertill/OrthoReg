"""Library-coupled PINN / Universal ODE baselines (``model.physics=library``).

The torch library is checked against ``feature_library.transform``; the
baselines' physics terms are recomputed independently from the PySINDy
features and the raw parameters.
"""

import numpy as np
import pysindy as ps
import pytest
import torch
from omegaconf import OmegaConf

from orthoreg.models.baselines.library_torch import TorchFeatureLibrary
from orthoreg.models.baselines.pinn import PINNExperiment
from orthoreg.models.baselines.universal_ode import UniversalODEExperiment

MU = 0.012


def _fitted_library(n_states, degree, n_frequencies, seed=0):
    feature_library = ps.ConcatLibrary([
        ps.PolynomialLibrary(degree=degree, include_bias=False),
        ps.FourierLibrary(n_frequencies=n_frequencies),
    ])
    rng = np.random.default_rng(seed)
    feature_library.fit(rng.normal(size=(20, n_states)))
    return feature_library


@pytest.mark.parametrize("n_states,degree,n_frequencies", [(2, 2, 1), (3, 3, 2)])
def test_torch_library_matches_pysindy_transform(n_states, degree, n_frequencies):
    feature_library = _fitted_library(n_states, degree, n_frequencies)
    library = TorchFeatureLibrary.from_feature_library(feature_library)
    assert library.feature_names == list(feature_library.get_feature_names())
    assert library.n_states == n_states

    x = np.random.default_rng(1).normal(scale=2.0, size=(50, n_states))
    expected = np.asarray(feature_library.transform(x))
    got = library(torch.tensor(x, dtype=torch.float64)).numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_torch_library_keeps_batch_dims_and_gradients():
    library = TorchFeatureLibrary.from_feature_library(_fitted_library(2, 2, 1))
    x = torch.randn(3, 5, 2, requires_grad=True)
    out = library(x)
    assert out.shape == (3, 5, library.n_features)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("names", [["x0", "exp(x0)"], ["1", "x0"], ["x0 "], ["tanh(x1)"]])
def test_torch_library_rejects_unknown_feature_names(names):
    with pytest.raises(ValueError, match="Unrecognized library feature name"):
        TorchFeatureLibrary(names)


def _make_cfg(physics, **model):
    return OmegaConf.create({
        "dataset": {"dataset_name": "pendulum", "n_out_of_library_terms": 3},
        "model": {"hidden_dim": 8, "num_layers": 3, "activation": "tanh",
                  "physics": physics, **model},
        "training": {"lr": 0.01, "symbolic_threshold": 0.045,
                     "orthogonal_symbolic_reg_weight": MU},
    })


def _batch(batch=3, seq_len=5, seed=0):
    torch.manual_seed(seed)
    y = torch.randn(batch, seq_len, 2)
    dy = torch.randn(batch, seq_len, 2)
    t = torch.linspace(0.0, 1.0, seq_len).expand(batch, -1)
    return y, None, dy, t


def _run_training_step(exp, batch):
    """One real Adam step through ``training_step`` without a Trainer."""
    optimizer = exp.configure_optimizers()["optimizer"]
    exp.optimizers = lambda: optimizer
    exp.manual_backward = lambda loss: loss.backward()
    exp.log = lambda *args, **kwargs: None
    exp.training_step(batch, 0)
    return optimizer


def _optimizer_holds(optimizer, param):
    return any(p is param for group in optimizer.param_groups for p in group["params"])


@pytest.mark.parametrize("Experiment", [PINNExperiment, UniversalODEExperiment])
def test_library_baseline_trains_coefficients(Experiment):
    feature_library = _fitted_library(2, 2, 1)
    exp = Experiment(_make_cfg("library"), feature_library)
    n_features = len(feature_library.get_feature_names())
    assert exp.coef_.shape == (n_features, 2)
    assert exp.l1_reg_weight == MU
    assert not hasattr(exp, "omega0") and not hasattr(exp, "alpha")

    y, _, dy, t = batch = _batch()
    flat = (t.reshape(-1, 1), y.reshape(-1, 2), dy.reshape(-1, 2))
    if Experiment is UniversalODEExperiment:
        flat = (t, y, dy)
    _, losses = exp.total_loss(*flat)
    torch.testing.assert_close(losses["l1_reg"].detach(), MU * exp.coef_.detach().abs().sum())

    before = exp.coef_.detach().clone()
    optimizer = _run_training_step(exp, batch)
    assert _optimizer_holds(optimizer, exp.coef_)
    assert exp.coef_.grad is not None and exp.coef_.grad.abs().sum() > 0
    assert not torch.equal(before, exp.coef_.detach())


def test_pinn_library_physics_residual():
    feature_library = _fitted_library(2, 2, 1)
    exp = PINNExperiment(_make_cfg("library"), feature_library)
    y, _, _, t = _batch()
    t, y = t.reshape(-1, 1), y.reshape(-1, 2)
    with torch.no_grad():
        theta = torch.tensor(np.asarray(feature_library.transform(y.numpy())), dtype=torch.float32)
        pred = exp.pinn_net(torch.cat([t, y], dim=1))
        expected = ((pred - theta @ exp.coef_) ** 2).mean()
        got = exp.physics_loss(t, y)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_uode_library_dynamics_has_no_identity_term():
    feature_library = _fitted_library(2, 2, 1)
    exp = UniversalODEExperiment(_make_cfg("library"), feature_library)
    y = _batch()[0].reshape(-1, 2)
    with torch.no_grad():
        theta = torch.tensor(np.asarray(feature_library.transform(y.numpy())), dtype=torch.float32)
        torch.testing.assert_close(exp.known_dynamics(None, y), theta @ exp.coef_, rtol=1e-5, atol=1e-6)
        exp.coef_.zero_()
        assert exp.known_dynamics(None, y).abs().max() == 0.0
        assert exp.known_dynamics(None, y[0]).shape == (1, 2)


@pytest.mark.parametrize("Experiment,loss_keys", [
    (PINNExperiment, {"total_loss", "data_loss", "physics_loss", "bc_loss"}),
    (UniversalODEExperiment, {"total_loss", "data_loss", "trajectory_loss", "physics_reg"}),
])
def test_known_physics_is_unchanged(Experiment, loss_keys):
    exp = Experiment(_make_cfg("known"), _fitted_library(2, 2, 1))
    assert exp.physics == "known"
    assert not hasattr(exp, "coef_")
    assert exp.omega0.item() == 1.0 and exp.alpha.item() == pytest.approx(0.2)
    net = exp.pinn_net if Experiment is PINNExperiment else exp.uode_net
    params = {id(p) for p in exp.parameters()}
    assert params == {id(p) for p in net.parameters()} | {id(exp.omega0), id(exp.alpha)}

    y, _, dy, t = batch = _batch()
    flat = (t.reshape(-1, 1), y.reshape(-1, 2), dy.reshape(-1, 2))
    if Experiment is UniversalODEExperiment:
        flat = (t, y, dy)
    _, losses = exp.total_loss(*flat)
    assert set(losses) == loss_keys
    optimizer = _run_training_step(exp, batch)
    assert _optimizer_holds(optimizer, exp.omega0)


def test_default_physics_is_known():
    cfg = _make_cfg("known")
    del cfg.model["physics"]
    assert PINNExperiment(cfg).physics == "known"
    assert UniversalODEExperiment(cfg).physics == "known"
    with pytest.raises(ValueError, match="model.physics"):
        PINNExperiment(_make_cfg("unknown"))


@pytest.mark.parametrize("Experiment,default", [(PINNExperiment, 1.0), (UniversalODEExperiment, 0.1)])
def test_physics_weight_is_read_from_model_config(Experiment, default):
    assert Experiment(_make_cfg("known", physics_weight=0.5)).physics_weight == 0.5
    cfg = _make_cfg("known")
    cfg.training.physics_weight = 0.3
    assert Experiment(cfg).physics_weight == 0.3
    assert Experiment(_make_cfg("known")).physics_weight == default
