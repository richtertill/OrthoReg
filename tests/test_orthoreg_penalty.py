"""Unit tests for the training-time OrthoReg penalty."""

import pytest
import torch

from orthoreg.regularization.orthoreg import orthoreg_penalty


class TestOrthoregPenalty:
    """Tests for orthoreg.regularization.orthoreg.orthoreg_penalty."""

    def test_returns_scalar(self):
        pred_aug = torch.randn(2, 10, 3)
        symbolic_basis = torch.randn(2, 10, 5, 3)
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        assert penalty.shape == torch.Size([])
        assert penalty.ndim == 0

    def test_zero_when_neural_output_is_zero(self):
        pred_aug = torch.zeros(4, 8, 2)
        symbolic_basis = torch.randn(4, 8, 6, 2)
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        torch.testing.assert_close(penalty, torch.tensor(0.0))

    def test_zero_when_basis_terms_are_zero(self):
        pred_aug = torch.randn(3, 5, 2)
        symbolic_basis = torch.zeros(3, 5, 4, 2)
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        torch.testing.assert_close(penalty, torch.tensor(0.0))

    def test_nonzero_when_components_overlap(self):
        torch.manual_seed(0)
        pred_aug = torch.randn(2, 6, 1)
        # Each basis term is a scaled copy of pred_aug -> nonzero inner products.
        symbolic_basis = pred_aug.unsqueeze(2).expand(2, 6, 3, 1) * 0.5
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        assert penalty.item() > 0.0

    def test_gradient_flows_to_neural_output(self):
        pred_aug = torch.randn(2, 4, 2, requires_grad=True)
        symbolic_basis = torch.randn(2, 4, 3, 2)
        penalty = orthoreg_penalty(pred_aug, symbolic_basis)
        penalty.backward()
        assert pred_aug.grad is not None
        assert torch.isfinite(pred_aug.grad).all()

    def test_invalid_pred_aug_shape_raises(self):
        pred_aug = torch.randn(2, 10)
        symbolic_basis = torch.randn(2, 10, 5, 3)
        with pytest.raises(ValueError, match="pred_aug must be"):
            orthoreg_penalty(pred_aug, symbolic_basis)

    def test_invalid_symbolic_basis_shape_raises(self):
        pred_aug = torch.randn(2, 10, 3)
        symbolic_basis = torch.randn(2, 10, 5)
        with pytest.raises(ValueError, match="symbolic_basis must be"):
            orthoreg_penalty(pred_aug, symbolic_basis)
