"""Shape and wiring tests for the hybrid Forecaster + HybridExperiment stack."""

import pytest
import pysindy as ps
import torch
from omegaconf import OmegaConf

from orthoreg.models.exp import HybridExperiment
from orthoreg.models.forecasters import Forecaster
from orthoreg.models.networks import MLP, PhysicalModel
from orthoreg.regularization.orthoreg import orthoreg_penalty


def _make_hybrid_experiment(n_states=2, batch=2, seq_len=8):
    """Build a minimal hybrid model + experiment for shape checks."""
    feature_library = ps.ConcatLibrary([
        ps.PolynomialLibrary(degree=2, include_bias=False),
        ps.FourierLibrary(n_frequencies=1),
    ])
    states_np = torch.randn(batch, seq_len, n_states).numpy()
    t = torch.linspace(0.0, 1.0, seq_len).numpy()
    feature_library.fit(states_np, t)
    n_features = feature_library.n_output_features_

    cfg = OmegaConf.create({
        "dataset": {
            "dataset_name": "pendulum",
            "forecaster": {"out_dim": n_states},
        },
        "model": {
            "hybrid_setting": True,
            "symbolic_regression": "sindy",
        },
        "training": {
            "regularization": "orthogonal",
            "symbolic_threshold": 0.045,
            "l2_node_reg_weight": 0.005,
            "l2_symbolic_reg_weight": 0.001,
            "orthogonal_node_reg_weight": 0.005,
            "orthogonal_symbolic_reg_weight": 0.003,
            "lr": 0.01,
            "freeze_neural": False,
            "freeze_epochs": 0,
        },
    })

    device = torch.device("cpu")
    model_phy = PhysicalModel(
        feature_library=feature_library,
        device=device,
        cfg=cfg,
        n_states=n_states,
    )
    model_aug = MLP(state_c=n_features, hidden=32, num_layers=2, out_c=n_states)
    net = Forecaster(
        model_phy=model_phy,
        model_aug=model_aug,
        hybrid_setting=True,
    )
    exp = HybridExperiment(net, cfg)
    transformed_y = torch.randn(batch, seq_len, n_features)
    return exp, transformed_y, n_states, n_features


class TestHybridForward:
    """Verify the shipping hybrid model produces expected tensor shapes."""

    def test_derivative_forward_shapes(self):
        exp, transformed_y, n_states, _ = _make_hybrid_experiment()
        pred_phy, pred_aug = exp.derivative_forward(transformed_y)
        assert pred_phy.shape == transformed_y.shape[:2] + (n_states,)
        assert pred_aug.shape == transformed_y.shape[:2] + (n_states,)

    def test_symbolic_basis_shape(self):
        exp, transformed_y, n_states, n_features = _make_hybrid_experiment()
        symbolic_basis = exp.get_symbolic_basis_predictions(transformed_y)
        assert symbolic_basis.shape == (
            transformed_y.shape[0],
            transformed_y.shape[1],
            n_features,
            n_states,
        )

    def test_orthoreg_penalty_wires_through_experiment_outputs(self):
        exp, transformed_y, _, _ = _make_hybrid_experiment()
        _, pred_aug = exp.derivative_forward(transformed_y)
        symbolic_basis = exp.get_symbolic_basis_predictions(transformed_y)
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        assert penalty.ndim == 0
        assert torch.isfinite(penalty)

    def test_derivative_loss_keys(self):
        exp, transformed_y, _, _ = _make_hybrid_experiment()
        dy = torch.randn_like(transformed_y[..., :2])
        losses = exp.get_derivative_loss(transformed_y, dy, hybrid_setting=True)
        assert set(losses.keys()) == {"total_loss", "phy_loss", "aug_loss", "pred_dy"}
        assert losses["aug_loss"].ndim == 0
