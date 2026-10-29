"""Known-answer tests for the support-recovery metrics."""

import numpy as np
import pytest
import torch

from orthoreg.models.experiment import support_metrics

# 4 library features x 2 state dims = 8 slots, threshold 0.045.
W_TRUE = np.array([[1.0, 0.0],
                   [0.0, -2.0],
                   [0.5, 0.0],
                   [0.0, 0.0]])      # P = 3 true in-library slots, 5 negatives
W_HAT = np.array([[0.9, 0.0],       # TP
                  [0.0, 0.0],       # FN at (1, 1)
                  [0.6, 0.1],       # TP at (2, 0), FP at (2, 1)
                  [0.01, 0.3]])     # below threshold at (3, 0), FP at (3, 1)
# TP = 2, FP = 2, FN = 1, nnz = 4


def test_hand_built_example():
    m = support_metrics(W_HAT, W_TRUE, threshold=0.045, n_out_of_library_terms=2)
    assert (m["true_pos"], m["false_pos"], m["false_neg"], m["num_nonzero"]) == (2, 2, 1, 4)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7)
    assert m["fpr"] == pytest.approx(2 / 5)
    assert m["full_recall"] == pytest.approx(2 / 5)    # TP / (P + K)
    assert m["full_f1"] == pytest.approx(4 / 9)        # 2TP / (nnz + P + K)


def test_f1_is_harmonic_mean_of_precision_and_recall():
    m = support_metrics(W_HAT, W_TRUE, threshold=0.045)
    p, r = m["precision"], m["recall"]
    assert m["f1"] == pytest.approx(2 * p * r / (p + r))


def test_without_out_of_library_terms_full_scores_equal_in_library_scores():
    m = support_metrics(W_HAT, W_TRUE, threshold=0.045, n_out_of_library_terms=0)
    assert m["full_recall"] == pytest.approx(m["recall"])
    assert m["full_f1"] == pytest.approx(m["f1"])


def test_perfect_recovery():
    m = support_metrics(W_TRUE * 1.1, W_TRUE, threshold=0.045, n_out_of_library_terms=3)
    assert (m["precision"], m["recall"], m["f1"], m["fpr"]) == (1.0, 1.0, 1.0, 0.0)
    assert m["full_recall"] == pytest.approx(3 / 6)
    assert m["full_f1"] == pytest.approx(6 / 9)


def test_threshold_is_strict():
    # |0.045| is not above the threshold, |0.046| is.
    pred = np.array([[0.045, 0.0], [0.0, 0.046]])
    true = np.array([[1.0, 0.0], [0.0, 1.0]])
    m = support_metrics(pred, true, threshold=0.045)
    assert m["true_pos"] == 1 and m["false_neg"] == 1


def test_empty_prediction_scores_zero_without_nan():
    m = support_metrics(np.zeros_like(W_TRUE), W_TRUE, threshold=0.045, n_out_of_library_terms=2)
    for key in ("precision", "recall", "f1", "fpr", "full_recall", "full_f1"):
        assert m[key] == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        support_metrics(np.zeros((3, 2)), np.zeros((4, 2)), threshold=0.045)


def test_compute_dynamics_metrics_uses_dataset_k():
    """HybridExperiment.compute_dynamics_metrics maps the W_true dict onto
    the coefficient layout and uses ``dataset.n_out_of_library_terms``."""
    from tests.test_experiment import _make_experiment

    exp, _, _ = _make_experiment("orthogonal", hybrid=True)   # K = 3 in that cfg
    exp.log = lambda *args, **kwargs: None
    names = exp.net.model_phy.feature_names
    pred = torch.zeros(len(names), 2)
    pred[names.index("x1"), 0] = 1.0       # TP
    pred[names.index("x0^2"), 1] = 0.5     # FP
    true = {name: torch.zeros(2) for name in names}
    true["x1"][0] = 1.0
    true["x1"][1] = -0.2                   # FN
    true["sin(1 x0)"][1] = -1.0            # FN

    m = exp.compute_dynamics_metrics(pred, true, names, batch_idx=1)
    n_slots = 2 * len(names)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 / 5)
    assert m["fpr"] == pytest.approx(1 / (n_slots - 3))
    assert m["full_recall"] == pytest.approx(1 / 6)
    assert m["full_f1"] == pytest.approx(2 / 8)
