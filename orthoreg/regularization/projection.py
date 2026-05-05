"""Analytical projection helpers for examples / tests.

These are the offline counterparts to the training-time OrthoReg penalty
in :mod:`orthoreg.regularization.orthoreg`. Given a finite symbolic
library and a candidate function ``g`` evaluated on observed states, they
build the empirical Gram matrix, project ``g`` onto the span of the
library, and return the orthogonal residual ``g - P_V g``.

The training loop does *not* use these helpers (it computes the
empirical-orthogonality penalty directly inside the autograd graph; see
:func:`orthoreg.regularization.orthoreg.orthoreg_penalty`). They exist so
that ``examples/simple_pendulum_example.py`` and the unit tests can verify
the geometry of the projection that OrthoReg minimises.
"""

from typing import Tuple

import numpy as np


def compute_empirical_inner_product(f: np.ndarray, h: np.ndarray) -> float:
    """Empirical inner product <f, h> = (1/T) sum_t f(z_t) * h(z_t)."""
    return float(np.mean(f * h))


def compute_feature_gram_matrix(feature_library, trajectories: np.ndarray,
                                timepoints: np.ndarray):
    """Empirical Gram matrix G_ij = <phi_i, phi_j> over the given trajectories.

    Returns ``(G, feature_names)`` where ``G`` has shape
    ``(n_features, n_features)``.
    """
    feature_library.fit(trajectories, timepoints)
    feature_names = feature_library.get_feature_names()
    n_features = len(feature_names)

    feature_values = feature_library.transform(trajectories)

    G = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            G[i, j] = compute_empirical_inner_product(
                feature_values[:, :, i].flatten(),
                feature_values[:, :, j].flatten(),
            )
    return G, feature_names


def project_onto_feature_space(g_values: np.ndarray, feature_library,
                               trajectories: np.ndarray, timepoints: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose ``g`` into ``(P_V g, g - P_V g)`` against the library span."""
    G, feature_names = compute_feature_gram_matrix(
        feature_library, trajectories, timepoints
    )
    feature_values = feature_library.transform(trajectories)

    inner_products = np.zeros(len(feature_names))
    for k in range(len(feature_names)):
        inner_products[k] = compute_empirical_inner_product(
            feature_values[:, :, k].flatten(), g_values.flatten()
        )

    try:
        coeffs = np.linalg.solve(G, inner_products)
    except np.linalg.LinAlgError:
        # Singular Gram (e.g. linearly dependent library) -> fall back to
        # the pseudo-inverse so the projection is still well-defined.
        coeffs = np.linalg.pinv(G) @ inner_products

    projection = np.zeros_like(g_values)
    for j, c in enumerate(coeffs):
        projection += c * feature_values[:, :, j]

    residual = g_values - projection
    return projection, residual


def orthogonalize_function(g_values: np.ndarray, feature_library,
                           trajectories: np.ndarray, timepoints: np.ndarray
                           ) -> np.ndarray:
    """Convenience wrapper that returns only the orthogonal residual."""
    _, residual = project_onto_feature_space(
        g_values, feature_library, trajectories, timepoints
    )
    return residual
