"""OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems.

OrthoReg adds an explicit empirical-orthogonality penalty between a
symbolic library and a neural augmentation, so the neural component does
not absorb terms the symbolic library can already express.

Two surfaces are exposed:

- The training pipeline, configured with Hydra and run from the command
  line (see the README quickstart and ``configs/``).
- The orthogonal-projection utilities used inside the training loop,
  available directly from this package for analytical work or unit
  testing:

    >>> from orthoreg import (
    ...     compute_empirical_inner_product,
    ...     project_onto_feature_space,
    ...     orthogonalize_function,
    ... )

The full API is documented in the README and exercised by ``tests/``.
"""

__version__ = "0.1.0"

try:
    from orthoreg.regularization.orthoreg import (
        l2_penalty,
        orthoreg_penalty,
    )
    from orthoreg.regularization.projection import (
        compute_empirical_inner_product,
        compute_feature_gram_matrix,
        orthogonalize_function,
        project_onto_feature_space,
    )
except ImportError:
    # Allow ``import orthoreg`` to succeed before optional deps such as
    # pysindy are installed (used by the smoke test on a minimal env).
    pass

__all__ = [
    "orthoreg_penalty",
    "l2_penalty",
    "compute_empirical_inner_product",
    "compute_feature_gram_matrix",
    "project_onto_feature_space",
    "orthogonalize_function",
]
