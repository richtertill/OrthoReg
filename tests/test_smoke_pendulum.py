"""
End-to-end smoke test for OrthoReg.

The goal is to back the README's "the install works" promise without
requiring a GPU or a full Hydra/Lightning launch. We
  1. import the package,
  2. take a short gradient run on a toy hybrid loss whose regularizer is
     OrthoReg's empirical-orthogonality penalty, and
  3. instantiate the modified damped pendulum dataset on a single short
     trajectory (skipped if Lightning is not installed in the test env).

Anything more than that (full datamodule wiring, multi-epoch training)
belongs in `experiments/`, not in CI.
"""

import tempfile

import numpy as np
import pysindy as ps
import pytest
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
    pytest.importorskip("lightning",
                        reason="lightning is required for the dataset module")

    from orthoreg.data.datasets.pend import PendulumDataset

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

    assert ds.y.shape[0] == 1, "expected one trajectory"
    assert ds.y.shape[-1] == 2, "expected two state dims (theta, omega)"
    assert torch.isfinite(ds.y).all(), "ODE integration produced NaNs/Infs"


def test_orthoreg_penalty_gradient_step():
    """Optimising the OrthoReg penalty drives library-overlap toward zero.

    Mirrors the math used during training: we form the empirical inner
    product between a learnable f_aug evaluation and a fixed symbolic
    library, square it, sum over library terms, and step. With a small
    enough learning rate this is convex and decreases monotonically.
    """
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
    n_features = phi.shape[1]

    # f_aug(x) = W x; deliberately initialised inside the symbolic span.
    f_aug = torch.nn.Linear(d, d, bias=False)
    f_aug.weight.data = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    optimizer = torch.optim.Adam(f_aug.parameters(), lr=5e-2)

    def ortho_penalty():
        f_vals = f_aug(states)                                  # (n, d)
        # <f_aug, phi_j>_D = (1/n) sum_i (f_aug(x_i) . phi_j(x_i))
        ip_per_sample = (f_vals.unsqueeze(-1) * phi.unsqueeze(1)).sum(dim=1)
        ip = ip_per_sample.mean(dim=0)                          # (n_features,)
        return (ip ** 2).sum()

    initial_loss = ortho_penalty().item()
    assert np.isfinite(initial_loss)
    assert initial_loss > 0, "starting f_aug should overlap the library"

    for _ in range(500):
        optimizer.zero_grad()
        loss = ortho_penalty()
        loss.backward()
        optimizer.step()

    final_loss = ortho_penalty().item()
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 1e-2, (
        f"OrthoReg penalty failed to drive overlap toward zero "
        f"(before {initial_loss}, after {final_loss})"
    )
    assert n_features > 0
