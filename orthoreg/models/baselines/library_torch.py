"""Differentiable torch evaluation of the PySINDy symbolic library.

The library-coupled PINN / Universal ODE baselines (``model.physics=library``)
use the same features ``Theta(x)`` as OrthoReg, but need them inside the
autograd graph. ``TorchFeatureLibrary`` rebuilds every feature from its
PySINDy name, in the order of ``feature_library.get_feature_names()``, so
``TorchFeatureLibrary(...)(x)`` matches ``feature_library.transform(x)``.

Supported names are products of polynomial factors (``x0``, ``x0^2``,
``x0 x1``) and Fourier factors (``sin(1 x0)``, ``cos(2 x1)``), i.e. the
output of ``PolynomialLibrary(include_bias=False)`` and ``FourierLibrary``.
"""

import re

import numpy as np
import torch

from orthoreg.models.experiment import support_metrics

# One factor of a feature name: ``sin(k xi)`` / ``cos(k xi)`` or ``xi`` / ``xi^p``.
_FACTOR = re.compile(r"(sin|cos)\((\d+) x(\d+)\)|x(\d+)(?:\^(\d+))?")


def parse_feature_name(name):
    """Split a PySINDy feature name into ``(kind, state_index, arg)`` factors.

    ``kind`` is ``'pow'`` (``arg`` = exponent), ``'sin'`` or ``'cos'``
    (``arg`` = frequency). Raises ``ValueError`` on any other name.
    """
    factors, pos = [], 0
    while pos < len(name):
        match = _FACTOR.match(name, pos)
        if match is None:
            break
        if match.group(1):
            factors.append((match.group(1), int(match.group(3)), int(match.group(2))))
        else:
            factors.append(("pow", int(match.group(4)), int(match.group(5) or 1)))
        pos = match.end()
        if pos < len(name):
            if name[pos] != " ":
                break
            pos += 1
    if not factors or pos != len(name) or name.endswith(" "):
        raise ValueError(
            f"Unrecognized library feature name {name!r}: only polynomial terms "
            "without bias ('x0', 'x0^2', 'x0 x1') and Fourier terms "
            "('sin(1 x0)', 'cos(1 x1)') are supported."
        )
    return factors


class TorchFeatureLibrary:
    """Torch version of a fitted PySINDy library, built from its feature names."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.terms = [parse_feature_name(name) for name in self.feature_names]
        self.n_features = len(self.terms)
        self.n_states = 1 + max(i for factors in self.terms for _, i, _ in factors)

    @classmethod
    def from_feature_library(cls, feature_library):
        return cls(feature_library.get_feature_names())

    def __call__(self, x):
        """Evaluate ``Theta(x)``: ``(..., n_states) -> (..., n_features)``."""
        columns = []
        for factors in self.terms:
            column = torch.ones_like(x[..., 0])
            for kind, i, arg in factors:
                if kind == "pow":
                    column = column * x[..., i] ** arg
                elif kind == "sin":
                    column = column * torch.sin(arg * x[..., i])
                else:
                    column = column * torch.cos(arg * x[..., i])
            columns.append(column)
        return torch.stack(columns, dim=-1)


def library_support_metrics(coef, w_true, feature_names, threshold, n_out_of_library_terms=0):
    """Support metrics of ``coef`` against the dataset's ``W_true`` dict.

    Same construction as ``HybridExperiment.compute_dynamics_metrics``:
    features missing from ``w_true`` have a zero true coefficient.
    """
    pred = coef.detach().cpu().numpy()
    true = np.zeros_like(pred)
    for i, feature in enumerate(feature_names):
        if feature in w_true:
            true[i] = torch.as_tensor(w_true[feature]).cpu().numpy()
    return support_metrics(pred, true, threshold, n_out_of_library_terms)
