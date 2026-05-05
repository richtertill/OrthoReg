"""Regularization layer for OrthoReg.

Two distinct surfaces live here:

- :mod:`orthoreg.regularization.orthoreg` -- the *training-time*
  empirical-orthogonality penalty. ``HybridExperiment`` calls
  :func:`orthoreg_penalty` from inside its derivative-loss step; this is
  the file the paper's main contribution lives in.
- :mod:`orthoreg.regularization.projection` -- the *offline* analytical
  projection helpers used by ``examples/`` and ``tests/`` to verify the
  geometry of the penalty (Gram matrix, ``P_V g``, ``g - P_V g``).

Backwards-compatible aliases for the offline helpers are re-exported here
so that user code can ``from orthoreg.regularization import
project_onto_feature_space`` without caring about the file split.
"""

from orthoreg.regularization.orthoreg import l2_penalty, orthoreg_penalty
from orthoreg.regularization.projection import (
    compute_empirical_inner_product,
    compute_feature_gram_matrix,
    orthogonalize_function,
    project_onto_feature_space,
)

__all__ = [
    # training-time
    "orthoreg_penalty",
    "l2_penalty",
    # offline / analytical
    "compute_empirical_inner_product",
    "compute_feature_gram_matrix",
    "project_onto_feature_space",
    "orthogonalize_function",
]
