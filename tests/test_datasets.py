"""Generation, shape and ground-truth tests for the five paper systems."""

import numpy as np
import pytest
import torch

from orthoreg.data.datamodule import setup_dataloaders
from orthoreg.data.datasets import (
    DuffingOscillatorDataset,
    HelmholtzOscillatorDataset,
    LotkaVolterraDataset,
    PendulumDataset,
    SIREpidemicDataset,
)
from orthoreg.feature_library import setup_feature_library

N_SAMPLES, TIMES, GRANULARITY = 5, 2, 10
N_T = TIMES * GRANULARITY
N_T_EXTRA = int(N_T * 1.3)

# dataset -> (state dim, K, {(feature, state dim): true coefficient})
EXPECTED = {
    "pendulum": (2, 3, {("x1", 0): 1.0, ("sin(1 x0)", 1): -1.0, ("x1", 1): -0.2}),
    "duffing": (2, 1, {("x1", 0): 1.0, ("x1", 1): -0.5, ("x0", 1): 1.0}),
    "lotka_volterra": (2, 2, {("x0", 0): 5.0, ("x0 x1", 0): -2.5,
                              ("x0 x1", 1): 2.5, ("x1", 1): -5.0}),
    "sir": (3, 3, {("x0 x1", 0): -2.0, ("x0 x1", 1): 2.0,
                   ("x1", 1): -1.0, ("x1", 2): 1.0}),
    "helmholtz": (2, 1, {("x1", 0): 1.0, ("x1", 1): -0.06,
                         ("x0", 1): -1.6, ("x0^2", 1): -0.16}),
}

SPLITS = ["train_data", "id_test_data", "ood_test_data_t2", "ood_test_data_t3"]


def _support(W_true):
    return {(name, k): float(value[k])
            for name, value in W_true.items()
            for k in range(len(value)) if float(value[k]) != 0.0}


@pytest.fixture(params=sorted(EXPECTED))
def generated(request, compose_cfg):
    name = request.param
    cfg = compose_cfg([
        f"dataset={name}",
        f"dataset.forecaster.n_samples={N_SAMPLES}",
        f"dataset.forecaster.times={TIMES}",
        f"dataset.forecaster.granularity={GRANULARITY}",
    ])
    feature_library = setup_feature_library(cfg)
    data_dict = setup_dataloaders(name, cfg, feature_library)
    return name, cfg, feature_library, data_dict


def test_shapes_and_finiteness(generated):
    name, _, _, data_dict = generated
    state_dim = EXPECTED[name][0]
    for split in SPLITS:
        for key, n_t in ((split, N_T), (split + "_extra", N_T_EXTRA)):
            ds = data_dict[key]
            assert ds.y.shape == (N_SAMPLES, n_t, state_dim), key
            assert ds.t.shape == (n_t,), key
            assert ds.dy["smooth"].shape == ds.y.shape, key
            assert torch.isfinite(ds.y).all(), key
            assert torch.isfinite(ds.dy["smooth"]).all(), key


def test_true_in_library_support(generated):
    name, _, _, data_dict = generated
    expected = EXPECTED[name][2]
    support = _support(data_dict["W_true"])
    assert set(support) == set(expected)
    for slot, value in expected.items():
        assert support[slot] == pytest.approx(value, rel=1e-6)
    # Every split shares the support (parameter shifts change values only).
    for split in SPLITS:
        assert set(_support(data_dict[split].W_true)) == set(expected), split


def test_number_of_out_of_library_terms(generated):
    name, cfg, _, _ = generated
    assert cfg.dataset.n_out_of_library_terms == EXPECTED[name][1]


def test_library_size(generated):
    name, _, feature_library, _ = generated
    state_dim = EXPECTED[name][0]
    # degree-2 polynomials without bias + sin/cos of every state
    n_poly = state_dim + state_dim * (state_dim + 1) // 2
    assert feature_library.n_output_features_ == n_poly + 2 * state_dim


def test_initial_conditions_are_sampled_from_the_box(generated):
    name, cfg, _, data_dict = generated
    y0 = data_dict["train_data"].y[:, 0, :].numpy()
    # Distinct draws, not the two corners of the box.
    assert len(np.unique(y0.round(6), axis=0)) == N_SAMPLES
    if name != "duffing":
        for k, (low, high) in enumerate(cfg.dataset.forecaster.y0_range):
            assert np.all(y0[:, k] >= low - 1e-6) and np.all(y0[:, k] <= high + 1e-6)


def test_ood_t3_scales_parameters(generated):
    name, _, _, data_dict = generated
    train = _support(data_dict["train_data"].W_true)
    ood = _support(data_dict["ood_test_data_t3"].W_true)
    if name in ("lotka_volterra", "sir"):
        for slot, value in train.items():
            assert ood[slot] == pytest.approx(1.2 * value, rel=1e-6)
    if name == "helmholtz":
        assert ood[("x0", 1)] == pytest.approx(1.2 * train[("x0", 1)], rel=1e-6)
        assert ood[("x0^2", 1)] == pytest.approx(1.2 * train[("x0^2", 1)], rel=1e-6)
        assert ood[("x1", 1)] == pytest.approx(train[("x1", 1)], rel=1e-6)


# --- W_true matches the in-library part of each vector field -----------------

def _rhs_without_missing_terms(name, x):
    """Vector field with the out-of-library strength set to zero."""
    if name == "pendulum":
        return PendulumDataset._pendulum_ode_func(x, 0.7, 1.0, 0.2, 0.0, 0.0, 0.0)
    if name == "duffing":
        return DuffingOscillatorDataset._duffing_ode_func(None, x, 0.7, -0.5, -1.0, 0.0)
    if name == "lotka_volterra":
        return LotkaVolterraDataset._lotka_volterra_ode_func(
            x, 0.7, 5.0, 2.5, 5.0, 2.5, 0.0, 0.0, 10.0, 2.0, 1e-6)
    if name == "sir":
        return SIREpidemicDataset._sir_ode_func(
            x, 0.7, 2.0, 1.0, 0.0, 0.0, 0.0, 3.0, 1.5, 2.0, 1e-6)
    return HelmholtzOscillatorDataset._helmholtz_ode_func(x, 0.7, 0.06, 1.6, 0.16, 0.0, 7.0)


def _rhs_with_missing_terms(name, x, t):
    if name == "pendulum":
        return PendulumDataset._pendulum_ode_func(x, t, 1.0, 0.2, 0.3, 0.25, 0.15)
    if name == "duffing":
        return DuffingOscillatorDataset._duffing_ode_func(None, x, t, -0.5, -1.0, 0.1)
    if name == "lotka_volterra":
        return LotkaVolterraDataset._lotka_volterra_ode_func(
            x, t, 5.0, 2.5, 5.0, 2.5, 3.0, 2.0, 10.0, 2.0, 1e-6)
    if name == "sir":
        return SIREpidemicDataset._sir_ode_func(
            x, t, 2.0, 1.0, 0.3, 0.25, 0.2, 3.0, 1.5, 2.0, 1e-6)
    return HelmholtzOscillatorDataset._helmholtz_ode_func(x, t, 0.06, 1.6, 0.16, 2.0, 7.0)


def _test_points(name, state_dim, n=20):
    rng = np.random.default_rng(0)
    if name == "sir":
        # Normalized population S + I + R = 1, where beta*S*I/N = beta*S*I.
        return rng.dirichlet(np.ones(3), size=n)
    return rng.uniform(0.2, 1.5, size=(n, state_dim))


def test_true_coefficients_reproduce_in_library_dynamics(generated):
    name, _, feature_library, data_dict = generated
    state_dim = EXPECTED[name][0]
    names = feature_library.get_feature_names()
    W = np.stack([data_dict["W_true"][f].numpy() for f in names]).astype(np.float64)
    points = _test_points(name, state_dim)
    phi = feature_library.transform(points)
    for x, phi_x in zip(points, phi):
        np.testing.assert_allclose(phi_x @ W, _rhs_without_missing_terms(name, x),
                                   rtol=1e-6, atol=1e-6)


def test_missing_dynamics_are_present(generated):
    name, _, feature_library, data_dict = generated
    state_dim = EXPECTED[name][0]
    names = feature_library.get_feature_names()
    W = np.stack([data_dict["W_true"][f].numpy() for f in names]).astype(np.float64)
    points = _test_points(name, state_dim)
    phi = feature_library.transform(points)
    residual = [np.abs(_rhs_with_missing_terms(name, x, 0.3) - phi_x @ W).max()
                for x, phi_x in zip(points, phi)]
    assert max(residual) > 1e-3


def test_test_loader_names_match_their_data(generated):
    """Loader ``i`` must hold the dataset that ``test_set_names[i]`` names.

    Metrics are labeled by these names, so a mismatch would silently report
    one split's error under another split's name.
    """
    from orthoreg.data.datamodule import HybridDataModule

    _, cfg, feature_library, data_dict = generated
    dm = HybridDataModule(cfg, data_dict, feature_library)
    dm.setup("test")
    loaders = dm.test_dataloader()

    assert dm.test_set_names == [
        "id_test_data", "ood_test_data_t2", "ood_test_data_t3",
        "id_test_data_extra", "ood_test_data_t2_extra", "ood_test_data_t3_extra",
    ]
    assert len(loaders) == len(dm.test_set_names)
    for name, loader in zip(dm.test_set_names, loaders):
        expected = torch.as_tensor(data_dict[name].y, dtype=torch.float32)
        loaded = loader.dataset.tensors[0].cpu().to(torch.float32)
        assert torch.equal(loaded, expected), name
