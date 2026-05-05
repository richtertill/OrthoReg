"""Self-contained illustration of the OrthoReg projection.

We synthesise a few short pendulum trajectories, build a polynomial
symbolic library, and check that the orthogonal residual of a candidate
neural-style function is empirically orthogonal to every library term.
This is the same projection mathematically that OrthoReg minimises during
training.
"""

import numpy as np
import pysindy as ps

from orthoreg.regularization import (
    compute_empirical_inner_product,
    orthogonalize_function,
    project_onto_feature_space,
)


print("OrthoReg example: damped pendulum + orthogonal projection")

# 1. Synthesise a couple of damped-pendulum trajectories.
t = np.linspace(0, 10, 100)
theta0 = [0.5, 1.0]
trajectories = []
for th0 in theta0:
    theta = th0 * np.exp(-0.05 * t) * np.cos(np.sqrt(1 - 0.05 ** 2) * t)
    trajectories.append(theta)
x = np.array(trajectories).reshape(len(theta0), len(t), 1)
print(f"[1/4] {len(theta0)} trajectories, {len(t)} timesteps each")

# 2. Build a polynomial symbolic library.
feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
feature_lib.fit(x, t)
feature_names = feature_lib.get_feature_names()
print(f"[2/4] polynomial library: {feature_names}")

# 3. Construct a "neural" candidate that overlaps the library on purpose.
neural_component = 0.3 * x[:, :, 0] + 0.1 * (x[:, :, 0] ** 2)
features = feature_lib.transform(x)
print("[3/4] inner products before orthogonalisation:")
for i, name in enumerate(feature_names):
    ip = compute_empirical_inner_product(
        neural_component.flatten(), features[:, :, i].flatten()
    )
    print(f"      <neural, {name}> = {ip:+.6f}")

# 4. Project out the library span and verify the residual is orthogonal.
orthogonal_component = orthogonalize_function(neural_component, feature_lib, x, t)
print("[4/4] inner products after orthogonalisation (expect ~0):")
for i, name in enumerate(feature_names):
    ip = compute_empirical_inner_product(
        orthogonal_component.flatten(), features[:, :, i].flatten()
    )
    print(f"      <orth, {name}> = {ip:+.2e}")

# Sanity checks: projection + residual reconstructs the original, and the
# residual is orthogonal to every library term up to numerical tolerance.
projection, residual = project_onto_feature_space(neural_component, feature_lib, x, t)
reconstruction_error = np.abs((projection + residual) - neural_component).max()
print(f"\nProjection + residual = original: max error = {reconstruction_error:.2e}")

max_ip = max(
    abs(compute_empirical_inner_product(residual.flatten(), features[:, :, i].flatten()))
    for i in range(len(feature_names))
)
print(f"Residual orthogonal to every library term: max |<r, phi>| = {max_ip:.2e}")

print(
    "\nL2 regularisation only controls ||f_aug||; OrthoReg additionally "
    "enforces <f_aug, phi_j> = 0 for every library term, i.e. the neural "
    "component cannot re-express what the symbolic library already captures."
)
