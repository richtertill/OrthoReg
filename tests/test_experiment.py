"""Wiring tests for HybridExperiment (Algorithm 1).

Every expected value is computed independently from the raw tensors
(features, coefficients, network output) rather than through the
experiment's own helpers.
"""

import pysindy as ps
import pytest
import torch
from omegaconf import OmegaConf

from orthoreg.models.experiment import HybridExperiment
from orthoreg.models.forecasters import Forecaster
from orthoreg.models.networks import MLP, PhysicalModel

L1_WEIGHT = {"orthogonal": 0.003, "l2": 0.001, "none": 0.0}
NODE_WEIGHT = {"orthogonal": 0.005, "l2": 0.005, "none": 0.0}


def _make_cfg(regularization, hybrid, node_weight=None):
    node = {"orthogonal": 0.005, "l2": 0.005}
    if node_weight is not None:
        node = {"orthogonal": node_weight, "l2": node_weight}
    return OmegaConf.create({
        "dataset": {"dataset_name": "pendulum", "n_out_of_library_terms": 3},
        "model": {"hybrid_setting": hybrid, "symbolic_regression": "sindy"},
        "training": {
            "regularization": regularization,
            "symbolic_threshold": 0.045,
            "l2_node_reg_weight": node["l2"],
            "l2_symbolic_reg_weight": 0.001,
            "orthogonal_node_reg_weight": node["orthogonal"],
            "orthogonal_symbolic_reg_weight": 0.003,
            "lr": 0.01,
            "gradient_clip_value": 1.0,
            "n_derivative_epochs": 100,
            "freeze_neural": False,
            "freeze_epochs": 0,
        },
    })


def _make_experiment(regularization="orthogonal", hybrid=True, node_weight=None,
                     n_states=2, batch=3, seq_len=5, seed=0):
    torch.manual_seed(seed)
    feature_library = ps.ConcatLibrary([
        ps.PolynomialLibrary(degree=2, include_bias=False),
        ps.FourierLibrary(n_frequencies=1),
    ])
    states = torch.randn(batch, seq_len, n_states)
    feature_library.fit(states.numpy(), torch.linspace(0.0, 1.0, seq_len).numpy())
    features = torch.tensor(
        feature_library.transform(states.reshape(-1, n_states).numpy()),
        dtype=torch.float32,
    ).reshape(batch, seq_len, -1)
    n_features = features.shape[-1]

    cfg = _make_cfg(regularization, hybrid, node_weight)
    model_phy = PhysicalModel(feature_library=feature_library, device=torch.device("cpu"),
                              cfg=cfg, n_states=n_states)
    model_aug = MLP(state_c=n_features, hidden=16, num_layers=3, out_c=n_states)
    net = Forecaster(model_phy=model_phy, model_aug=model_aug, hybrid_setting=hybrid)
    exp = HybridExperiment(net, cfg)
    dy = torch.randn(batch, seq_len, n_states)
    return exp, features, dy


def _expected_loss(exp, features, dy, regularization, hybrid):
    """Independent computation of data + mu ||w||_1 + L_reg."""
    with torch.no_grad():
        W = exp.net.model_phy.coef_.detach().clone()
        phy = torch.einsum("btf,fk->btk", features, W)
        aug = exp.net.model_aug(features) if hybrid else torch.zeros_like(phy)
        data = ((dy - phy - aug) ** 2).sum(dim=-1).mean()
        l1 = L1_WEIGHT[regularization] * W.abs().sum()

        phi = features.reshape(-1, features.shape[-1])
        f = aug.reshape(-1, aug.shape[-1])
        if regularization == "orthogonal":
            reg = sum(
                ((f[:, k] * phi[:, j]).mean()) ** 2
                for j in range(phi.shape[1]) for k in range(f.shape[1])
            )
            reg = NODE_WEIGHT[regularization] * reg
        elif regularization == "l2":
            reg = NODE_WEIGHT[regularization] * (f ** 2).sum(dim=1).mean()
        else:
            reg = torch.tensor(0.0)
    return data, l1, reg


class _NoOpOptimizer:
    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        pass


def _run_training_step(exp, features, dy):
    """Call ``training_step`` without a Trainer; gradients are left in place."""
    optimizer = _NoOpOptimizer()
    optimizer.params = list(exp.net.parameters())
    exp.optimizers = lambda: optimizer
    exp.manual_backward = lambda loss: loss.backward()
    exp.clip_gradients = lambda *args, **kwargs: None
    exp.log = lambda *args, **kwargs: None
    t = torch.zeros(features.shape[0], features.shape[1])
    batch = (None, features, dy, t)
    return exp.training_step(batch, 0)


CASES = [("orthogonal", True), ("l2", True), ("none", False)]


@pytest.mark.parametrize("regularization,hybrid", CASES)
def test_training_step_loss_is_data_plus_l1_plus_penalty(regularization, hybrid):
    exp, features, dy = _make_experiment(regularization, hybrid)
    data, l1, reg = _expected_loss(exp, features, dy, regularization, hybrid)
    loss = _run_training_step(exp, features, dy)
    torch.testing.assert_close(loss.detach(), data + l1 + reg, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("regularization,hybrid", CASES)
def test_objective_components(regularization, hybrid):
    exp, features, dy = _make_experiment(regularization, hybrid)
    data, l1, reg = _expected_loss(exp, features, dy, regularization, hybrid)
    losses = exp.compute_objective(features, dy)
    torch.testing.assert_close(losses["total_loss"].detach(), data, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(losses["l1_reg"].detach(), l1, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(losses["node_reg"].detach(), torch.as_tensor(reg, dtype=torch.float32),
                               rtol=1e-6, atol=1e-7)


def test_penalty_weights_are_the_configured_lambda_and_mu():
    for regularization, hybrid in CASES:
        exp, _, _ = _make_experiment(regularization, hybrid)
        assert exp.l1_reg_weight == L1_WEIGHT[regularization]
        assert exp.node_reg_weight == NODE_WEIGHT[regularization]


@pytest.mark.parametrize("regularization", ["orthogonal", "l2"])
def test_network_gets_gradient_from_data_term_alone(regularization):
    exp, features, dy = _make_experiment(regularization, hybrid=True, node_weight=0.0)
    assert exp.node_reg_weight == 0.0
    _run_training_step(exp, features, dy)
    grads = [p.grad for p in exp.net.model_aug.parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0


def test_prediction_is_plain_sum_of_symbolic_and_neural_parts():
    exp, features, dy = _make_experiment("orthogonal", hybrid=True)
    with torch.no_grad():
        expected = (torch.einsum("btf,fk->btk", features, exp.net.model_phy.coef_)
                    + exp.net.model_aug(features))
        pred_dy, pred_phy, pred_aug = exp.predict_derivative(features)
        losses = exp.get_derivative_loss(features, dy)
    torch.testing.assert_close(pred_dy, expected)
    torch.testing.assert_close(pred_dy, pred_phy + pred_aug)
    torch.testing.assert_close(losses["pred_dy"], expected)


def test_pure_symbolic_prediction_has_no_network():
    exp, features, dy = _make_experiment("none", hybrid=False)
    with torch.no_grad():
        pred_dy, pred_phy, pred_aug = exp.predict_derivative(features)
        expected = torch.einsum("btf,fk->btk", features, exp.net.model_phy.coef_)
    assert pred_aug is None
    torch.testing.assert_close(pred_dy, expected)
    assert exp.regularization_penalty(features, pred_aug).item() == 0.0


def test_orthogonal_penalty_ignores_coefficients_and_mask():
    exp, features, dy = _make_experiment("orthogonal", hybrid=True)
    first = exp.compute_objective(features, dy)["aug_loss"].item()
    with torch.no_grad():
        exp.net.model_phy.coef_.mul_(-7.0).add_(3.0)
        exp.net.model_phy.mask_.zero_()
    second = exp.compute_objective(features, dy)["aug_loss"].item()
    assert first == pytest.approx(second, rel=1e-6)
    assert first > 0.0


def test_retrain_step_fits_plain_sum_without_regularizers():
    exp, features, dy = _make_experiment("orthogonal", hybrid=True)
    exp.structure_fixed = True
    exp.first_stage = False
    exp.feature_mask = torch.ones_like(exp.net.model_phy.coef_).numpy()
    data, _, _ = _expected_loss(exp, features, dy, "orthogonal", True)
    loss = _run_training_step(exp, features, dy)
    torch.testing.assert_close(loss.detach(), data, rtol=1e-6, atol=1e-7)


def test_fix_support_keeps_support_during_refit():
    exp, features, dy = _make_experiment("orthogonal", hybrid=True)
    coef = exp.net.model_phy.coef_
    with torch.no_grad():
        # Alternate entries clearly above / below the 0.045 threshold
        pattern = torch.arange(coef.numel()).reshape(coef.shape) % 2 == 0
        coef.copy_(torch.where(pattern, torch.ones_like(coef), 0.01 * torch.ones_like(coef)))
    exp.fix_support()

    mask = torch.tensor(exp.feature_mask)
    assert torch.equal(mask, pattern.float())
    assert torch.equal(exp.net.model_phy.mask_, pattern.float())
    assert exp.structure_fixed and not exp.first_stage
    assert torch.all(coef[~pattern] == 0.0)
    num_nonzero = int((coef.abs() > exp.threshold).sum())

    optimizer = exp.configure_optimizers()["optimizer"]
    exp.optimizers = lambda: optimizer
    exp.manual_backward = lambda loss: loss.backward()
    exp.log = lambda *args, **kwargs: None
    active_before = coef.detach()[pattern].clone()
    aug_before = [p.detach().clone() for p in exp.net.model_aug.parameters()]
    t = torch.zeros(features.shape[0], features.shape[1])
    for _ in range(5):
        exp.training_step((None, features, dy, t), 0)
        assert torch.all(coef.grad[~pattern] == 0.0)
        assert torch.all(coef.detach()[~pattern] == 0.0)

    assert int((coef.abs() > exp.threshold).sum()) == num_nonzero
    assert not torch.equal(coef.detach()[pattern], active_before)
    assert any(not torch.equal(p.detach(), b)
               for p, b in zip(exp.net.model_aug.parameters(), aug_before))
