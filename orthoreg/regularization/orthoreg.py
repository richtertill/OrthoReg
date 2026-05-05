"""Core OrthoReg penalty used during hybrid-SINDy training.

This is the file the paper's main contribution lives in. ``HybridExperiment``
calls :func:`orthoreg_penalty` from inside its derivative-loss step, and the
returned scalar is multiplied by ``training.orthogonal_node_reg_weight`` /
``training.orthogonal_symbolic_reg_weight`` and added to the SINDy fit loss
(see :mod:`orthoreg.models.exp`).

Math
----
Let ``f_aug`` be the neural augmentation evaluated on a batch of states and
``{phi_j}`` the active symbolic basis functions weighted by their current
SINDy coefficients (i.e. each ``phi_j`` is an *individual* contribution
``c_j * varphi_j(x)`` to the symbolic prediction). The empirical inner
product over a minibatch of size ``B*T`` is

    <f_aug, phi_j>_D = (1/(B*T)) * sum_{b,t} f_aug(x_{b,t}) * phi_j(x_{b,t})

and the OrthoReg penalty (per output state) is

    L_ortho = sum_j <f_aug, phi_j>_D ** 2

summed across output dimensions. Driving ``L_ortho`` to zero pushes
``f_aug`` to be empirically orthogonal to every active library term, so the
neural component can only capture what the symbolic library cannot. This is
the function-space-orthogonality formulation in Section 3 of the paper.

The analytical batch projection ``P_V g`` and ``g - P_V g`` (used by
``examples/`` and ``tests/`` to verify the geometry) live in
:mod:`orthoreg.regularization.projection`.
"""

from __future__ import annotations

import torch


def orthoreg_penalty(pred_aug: torch.Tensor,
                     symbolic_basis: torch.Tensor) -> torch.Tensor:
    """Empirical-orthogonality penalty between ``pred_aug`` and the library.

    Parameters
    ----------
    pred_aug : torch.Tensor, shape ``[B, T, state_dim]``
        Per-step output of the neural augmentation evaluated on the batch.
    symbolic_basis : torch.Tensor, shape ``[B, T, n_features, state_dim]``
        Per-step contribution of each symbolic library term, already
        multiplied by its current SINDy coefficient (so inactive terms
        contribute zero). Produced by
        :meth:`orthoreg.models.exp.HybridExperiment.get_symbolic_basis_predictions`.

    Returns
    -------
    torch.Tensor
        Scalar tensor: ``sum_j sum_d <f_aug_d, phi_j>_D ** 2``.
    """
    if pred_aug.dim() != 3:
        raise ValueError(
            f"pred_aug must be [B, T, state_dim], got shape {tuple(pred_aug.shape)}"
        )
    if symbolic_basis.dim() != 4:
        raise ValueError(
            f"symbolic_basis must be [B, T, n_features, state_dim], "
            f"got shape {tuple(symbolic_basis.shape)}"
        )

    # Broadcast pred_aug over the n_features axis and take the empirical
    # inner product (mean over B and T) for each (feature, state-dim) pair.
    products = pred_aug.unsqueeze(2) * symbolic_basis
    inner_products = torch.mean(products, dim=[0, 1])
    return torch.sum(inner_products.pow(2))


def l2_penalty(pred_aug: torch.Tensor) -> torch.Tensor:
    """Vanilla L2 penalty on the neural augmentation. Baseline used by
    ``training.regularization=l2``: ``mean_b ||f_aug(x_b)||_2``."""
    return torch.mean(torch.norm(pred_aug, p=2, dim=-1))
