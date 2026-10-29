"""Core OrthoReg penalty used during hybrid-SINDy training.

This is the file the paper's main contribution lives in. ``HybridExperiment``
(see :mod:`orthoreg.models.experiment`) calls :func:`orthoreg_penalty` or
:func:`l2_penalty` inside its training step and multiplies the result by
the node regularization weight ``lambda``.

Math
----
Let ``f_aug`` be the neural augmentation evaluated on a batch of states
``x_1, ..., x_B`` (and, for trajectory batches, on every time step) and
``phi_1, ..., phi_M`` the *raw* features of the symbolic library. Each
library slot is a vector field ``phi_j e_k`` (feature ``j`` placed in
state dimension ``k``), so the empirical inner products are

    g_jk = <f_aug, phi_j e_k>_D = (1/B) * sum_i f_aug,k(x_i) * phi_j(x_i)

and the OrthoReg penalty (Algorithm 1 in the paper) is

    L_orth = lambda * sum_{j,k} g_jk ** 2.

The features enter unscaled: neither the current SINDy coefficients nor the
thresholding mask touch the penalty, so ``f_aug`` is pushed away from the
span of the whole library, not just from the currently active terms.
Driving ``L_orth`` to zero makes ``f_aug`` empirically orthogonal to every
library slot.

The analytical batch projection ``P_V g`` and ``g - P_V g`` (used by
``examples/`` and ``tests/`` to verify the geometry) live in
:mod:`orthoreg.regularization.projection`.
"""

from __future__ import annotations

import torch


def orthoreg_penalty(pred_aug: torch.Tensor,
                     features: torch.Tensor) -> torch.Tensor:
    """Empirical-orthogonality penalty ``sum_{j,k} <f_aug, phi_j e_k>_D^2``.

    Parameters
    ----------
    pred_aug : torch.Tensor, shape ``[..., state_dim]``
        Output of the neural augmentation on the batch, e.g. ``[B, state_dim]``
        or ``[B, T, state_dim]``.
    features : torch.Tensor, shape ``[..., n_features]``
        Raw library features ``phi_j(x)`` at the same points, with the same
        leading dimensions as ``pred_aug``.

    Returns
    -------
    torch.Tensor
        Scalar ``sum_{j,k} g_jk ** 2`` (without the weight ``lambda``), where
        ``g_jk`` is the mean of ``phi_j * f_aug,k`` over all leading
        (batch and time) dimensions.
    """
    if pred_aug.dim() < 2:
        raise ValueError(
            f"pred_aug must be [..., state_dim], got shape {tuple(pred_aug.shape)}"
        )
    if features.shape[:-1] != pred_aug.shape[:-1]:
        raise ValueError(
            f"features must share the leading dimensions of pred_aug, got "
            f"{tuple(features.shape)} and {tuple(pred_aug.shape)}"
        )

    # Flatten batch (and time) into a single sample axis of size N.
    pred_aug = pred_aug.reshape(-1, pred_aug.shape[-1])
    features = features.reshape(-1, features.shape[-1]).to(pred_aug.dtype)

    # g[j, k] = (1/N) sum_i phi_j(x_i) * f_aug,k(x_i)
    inner_products = features.T @ pred_aug / pred_aug.shape[0]
    return torch.sum(inner_products.pow(2))


def l2_penalty(pred_aug: torch.Tensor) -> torch.Tensor:
    """L2 baseline ``mean_i ||f_aug(x_i)||_2^2`` (the paper's L2-regularized objective).

    The squared norm is taken over the state dimension and averaged over all
    leading (batch and time) dimensions. Used by
    ``training.regularization=l2``.
    """
    return torch.mean(torch.sum(pred_aug.pow(2), dim=-1))
