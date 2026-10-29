"""
Lightweight smoke tests for OrthoReg.

  1. import the package,
  2. take a short gradient run on a toy problem whose loss is OrthoReg's
     empirical-orthogonality penalty, and
  3. instantiate the modified damped pendulum dataset on a single short
     trajectory.

Full training runs live in ``test_smoke_training.py``.
"""

import tempfile

import numpy as np
import pysindy as ps
import torch


def test_orthoreg_package_imports():
    """The top-level package and its public API must import cleanly."""
    import orthoreg
    from orthoreg.regularization.projection import (
        compute_empirical_inner_product,
        orthogonalize_function,
        project_onto_feature_space,
    )
    from orthoreg.regularization.orthoreg import (
        l2_penalty,
        orthoreg_penalty,
    )

    assert orthoreg.__version__
    assert callable(compute_empirical_inner_product)
    assert callable(orthogonalize_function)
    assert callable(project_onto_feature_space)
    assert callable(orthoreg_penalty)
    assert callable(l2_penalty)


def test_pendulum_dataset_one_trajectory():
    """The modified damped pendulum can produce a single trajectory."""
    from orthoreg.data.datasets.pendulum import PendulumDataset

    with tempfile.TemporaryDirectory() as tmp:
        t = np.linspace(0.0, 1.0, 50)
        ds = PendulumDataset(
            n_samples=1,
            t=t,
            y0=[(0.5, 0.5), (0.0, 0.0)],
            omega0=1.0,
            alpha=0.2,
            beta1=0.3,
            beta2=0.25,
            beta3=0.15,
            seed=0,
            root=tmp,
            reload=True,
        )

    assert ds.y.shape == (1, 50, 2)
    assert torch.isfinite(ds.y).all(), "ODE integration produced NaNs/Infs"
    np.testing.assert_allclose(ds.y[0, 0].numpy(), [0.5, 0.0], atol=1e-6)


def test_orthoreg_penalty_gradient_step():
    """Optimizing the OrthoReg penalty drives library overlap toward zero."""
    from orthoreg.regularization.orthoreg import orthoreg_penalty

    torch.manual_seed(0)
    np.random.seed(0)

    n, d = 64, 2
    states_np = np.random.randn(n, d).astype(np.float32)
    states = torch.tensor(states_np, dtype=torch.float32)

    # Fixed symbolic library Phi(x): polynomials up to degree 2 evaluated
    # on the observed states.
    library = ps.PolynomialLibrary(degree=2, include_bias=False)
    library.fit(states_np)
    phi = torch.tensor(library.transform(states_np), dtype=torch.float32)

    # f_aug(x) = W x, deliberately initialized inside the library span.
    f_aug = torch.nn.Linear(d, d, bias=False)
    f_aug.weight.data = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    optimizer = torch.optim.Adam(f_aug.parameters(), lr=5e-2)

    initial_loss = orthoreg_penalty(f_aug(states), phi).item()
    assert np.isfinite(initial_loss)
    assert initial_loss > 0, "starting f_aug should overlap the library"

    for _ in range(500):
        optimizer.zero_grad()
        loss = orthoreg_penalty(f_aug(states), phi)
        loss.backward()
        optimizer.step()

    final_loss = orthoreg_penalty(f_aug(states), phi).item()
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 1e-2, (
        f"OrthoReg penalty failed to drive overlap toward zero "
        f"(before {initial_loss}, after {final_loss})"
    )
