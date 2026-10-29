"""End-to-end smoke runs: every system x {pure, L2, OrthoReg} + STLSQ on CPU.

Tiny trajectory budgets and a few epochs; the point is that the full
pipeline (data generation, Lightning fit, stage-2 retraining, test metrics)
runs and returns finite numbers, not that the models are any good.
"""

import math

import pytest

from orthoreg.training.train import train
from orthoreg.training.train_baseline import train_baseline

SYSTEMS = ["pendulum", "duffing", "lotka_volterra", "sir", "helmholtz"]
METHODS = {
    "pure": ["model=sindy", "training.regularization=none"],
    "l2": ["model=hybrid_sindy", "training.regularization=l2"],
    "orthogonal": ["model=hybrid_sindy", "training.regularization=orthogonal"],
}


def _tiny(system):
    return [
        f"dataset={system}",
        "dataset.forecaster.n_samples=4",
        "dataset.forecaster.times=2",
        "dataset.forecaster.granularity=10",
        "training.batch_size=4",
        "training.n_derivative_epochs=3",
        "training.n_retrain_epochs=2",
        "training.plot_trajectories=false",
        "model.hidden_dim=8",
    ]


def _check_results(results):
    assert results, "no test results returned"
    for per_loader in results:
        for key, value in per_loader.items():
            if key.endswith(("/f1", "/full_f1", "/fpr", "/precision", "/recall")) or "/f1/" in key:
                assert 0.0 <= value <= 1.0, (key, value)
            if "derivative_mse" in key:
                assert math.isfinite(value), (key, value)
    keys = set().union(*results)
    assert any("full_f1" in key for key in keys)
    assert any("fpr" in key for key in keys)


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("system", SYSTEMS)
def test_hybrid_and_pure_training(system, method, compose_cfg):
    cfg = compose_cfg(_tiny(system) + METHODS[method])
    retrain_results, test_results = train(cfg)
    _check_results(test_results)
    _check_results(retrain_results)


@pytest.mark.parametrize("system", SYSTEMS)
def test_stlsq_baseline(system, compose_cfg):
    cfg = compose_cfg(_tiny(system) + ["model=stlsq"])
    retrain_results, test_results = train_baseline(cfg)
    _check_results(test_results)
    _check_results(retrain_results)


@pytest.mark.parametrize("physics", ["known", "library"])
@pytest.mark.parametrize("model", ["pinn", "universal_ode"])
def test_pinn_and_universal_ode_baselines(model, physics, compose_cfg):
    cfg = compose_cfg(_tiny("pendulum") + [f"model={model}", f"model.physics={physics}"])
    retrain_results, test_results = train_baseline(cfg)
    for results in (test_results, retrain_results):
        assert results, "no test results returned"
        keys = set().union(*results)
        assert any("/state_mse" in key for key in keys)
        # Support metrics of xi only exist for the library-coupled variant
        assert any("/f1" in key for key in keys) == (physics == "library")
        for per_loader in results:
            for key, value in per_loader.items():
                if "/f1" in key or "/precision" in key or "/recall" in key:
                    assert 0.0 <= value <= 1.0, (key, value)
