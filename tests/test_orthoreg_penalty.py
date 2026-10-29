"""Known-answer tests for the training-time OrthoReg and L2 penalties.

Notation: ``Phi[i, j] = phi_j(x_i)`` (raw library features) and
``F[i, k] = f_aug,k(x_i)``. The OrthoReg penalty is
``sum_{j,k} g_jk^2`` with ``g_jk = (1/N) sum_i Phi[i, j] * F[i, k]``,
i.e. ``sum_{j,k} <f_aug, phi_j e_k>_D^2``.
"""

import pytest
import torch

from orthoreg.regularization.orthoreg import l2_penalty, orthoreg_penalty

# Hand-computed example: B = 2 samples, 2 features, 2 state dims.
#   g = Phi^T F / 2 = [[1, 3], [2, 4]] / 2 = [[0.5, 1.5], [1.0, 2.0]]
#   penalty = 0.25 + 2.25 + 1.0 + 4.0 = 7.5
PHI = torch.tensor([[1.0, 2.0],
                    [3.0, 4.0]])
F_AUG = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])


def _penalty_by_loops(pred_aug, features):
    """Independent loop implementation of sum_{j,k} <f_aug, phi_j e_k>_D^2."""
    pred_aug = pred_aug.reshape(-1, pred_aug.shape[-1])
    features = features.reshape(-1, features.shape[-1])
    n_samples, n_states = pred_aug.shape
    total = 0.0
    for j in range(features.shape[1]):
        for k in range(n_states):
            # <f_aug, phi_j e_k>_D = (1/N) sum_i f_aug(x_i)^T (phi_j(x_i) e_k)
            inner = 0.0
            for i in range(n_samples):
                slot = torch.zeros(n_states, dtype=pred_aug.dtype)
                slot[k] = features[i, j]
                inner += float(pred_aug[i] @ slot)
            total += (inner / n_samples) ** 2
    return total


class TestOrthoregPenaltyKnownAnswers:

    def test_hand_computed_value(self):
        penalty = orthoreg_penalty(F_AUG, PHI)
        assert penalty.item() == pytest.approx(7.5, abs=1e-12)

    def test_time_axis_is_averaged_like_the_batch_axis(self):
        # The same 2 samples arranged as [B=1, T=2] or [B=2, T=1].
        assert orthoreg_penalty(F_AUG.unsqueeze(0), PHI.unsqueeze(0)).item() == pytest.approx(7.5)
        assert orthoreg_penalty(F_AUG.unsqueeze(1), PHI.unsqueeze(1)).item() == pytest.approx(7.5)

    def test_hand_computed_gradient(self):
        # d/dF sum g^2 = (2/N) Phi g = Phi @ [[0.5, 1.5], [1, 2]]
        pred_aug = F_AUG.clone().requires_grad_(True)
        orthoreg_penalty(pred_aug, PHI).backward()
        expected = torch.tensor([[2.5, 5.5],
                                 [5.5, 12.5]])
        torch.testing.assert_close(pred_aug.grad, expected)

    def test_matches_loop_definition_on_random_tensors(self):
        torch.manual_seed(0)
        pred_aug = torch.randn(3, 4, 2, dtype=torch.float64)
        features = torch.randn(3, 4, 5, dtype=torch.float64)
        expected = _penalty_by_loops(pred_aug, features)
        assert orthoreg_penalty(pred_aug, features).item() == pytest.approx(expected, rel=1e-10)

    def test_zero_when_f_aug_orthogonal_to_every_slot(self):
        # Columns of F are orthogonal to both feature columns over N = 4 samples.
        features = torch.tensor([[1.0, 1.0],
                                 [1.0, -1.0],
                                 [1.0, 1.0],
                                 [1.0, -1.0]])
        pred_aug = torch.tensor([[1.0, 0.0],
                                 [0.0, 1.0],
                                 [-1.0, 0.0],
                                 [0.0, -1.0]])
        assert orthoreg_penalty(pred_aug, features).item() == 0.0

    def test_positive_when_f_aug_in_library_span(self):
        torch.manual_seed(1)
        features = torch.randn(50, 3)
        weights = torch.tensor([[0.5, 0.0],
                                [0.0, -1.0],
                                [0.2, 0.3]])
        pred_aug = features @ weights
        penalty = orthoreg_penalty(pred_aug, features)
        # Lower bound: the (j=0, k=0) slot alone, <f_aug_0, phi_0>^2.
        g00 = (features[:, 0] * pred_aug[:, 0]).mean() ** 2
        assert penalty.item() >= g00.item() > 0.0

    def test_zero_when_neural_output_is_zero(self):
        assert orthoreg_penalty(torch.zeros(4, 8, 2), torch.randn(4, 8, 6)).item() == 0.0

    def test_mismatched_leading_dims_raise(self):
        with pytest.raises(ValueError, match="leading dimensions"):
            orthoreg_penalty(torch.randn(2, 10, 3), torch.randn(2, 9, 5))

    def test_one_dimensional_pred_aug_raises(self):
        with pytest.raises(ValueError, match="pred_aug must be"):
            orthoreg_penalty(torch.randn(10), torch.randn(10, 3))


class TestL2PenaltyKnownAnswers:

    def test_mean_squared_norm(self):
        # ||(1, 2)||^2 = 5, ||(3, 4)||^2 = 25 -> mean 15 (squared, not the norm)
        pred_aug = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert l2_penalty(pred_aug).item() == pytest.approx(15.0)

    def test_identity_example(self):
        assert l2_penalty(F_AUG).item() == pytest.approx(1.0)

    def test_time_axis(self):
        pred_aug = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [B=1, T=2, 2]
        assert l2_penalty(pred_aug).item() == pytest.approx(15.0)
