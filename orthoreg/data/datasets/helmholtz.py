"""Damped Helmholtz oscillator with high-frequency corrugation (paper App. F.4).

The dynamics

    x0' = x1
    x1' = -delta*x1 - k*x0 - q*x0^2 + A*cos(Omega*x0)

have an in-library core under the shared degree-2 polynomial + Fourier(1)
library: {x1} in eq0 and {x1, x0, x0^2} in eq1. The corrugation
A*cos(Omega*x0) with Omega far from the library's unit frequency cannot be
reconstructed from the library and must be carried by the neural residual.

The potential V(x0) = (k/2)*x0^2 + (q/3)*x0^3 has an escape barrier at
x0 = -k/q for k, q > 0, so initial conditions must stay well inside it.
"""

from collections import OrderedDict
from hashlib import sha1
import json
import os

import numpy as np
import torch
from scipy.integrate import odeint as scipy_odeint

from orthoreg.data.datasets.utils import SeriesDataset, as_ranges
from orthoreg.paths import DATA_DIR


class HelmholtzOscillatorDataset(SeriesDataset):
    """Damped Helmholtz oscillator with out-of-library corrugation.

    The standard split provides
        train, id_test, ood_t2, ood_t3,
    where OOD-T2 scales the initial-condition box by ``ood_t2_y0_scale``
    (larger amplitudes, same parameters) and OOD-T3 scales the core
    parameters ``(k, q)`` by ``ood_t3_core_scale`` (same initial
    conditions). Each split is also exposed on a longer extrapolation time
    grid (``*_extra``).

    Parameters
    ----------
    n_samples: int
        Number of trajectories.
    t: np.ndarray
        Time points used to integrate the ODE.
    y0: list
        Per-state initial-condition ranges.
    delta, k, q: float
        Damping, linear and quadratic stiffness (in-library core).
    A, Omega: float
        Amplitude and frequency of the out-of-library corrugation.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(-6.0, 4.0), (-2.4, 2.4)],
        delta=0.06,
        k=1.6,
        q=0.16,
        A=2.0,
        Omega=7.0,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({
            'delta': delta,
            'k': k,
            'q': q,
            'A': A,
            'Omega': Omega,
        })

        self.true_equation = {
            'x0': "x0' = x1",
            'x1': f"x1' = -{delta}*x1 - {k}*x0 - {q}*x0^2 + {A}*cos({Omega}*x0)",
        }

        os.makedirs(root, exist_ok=True)
        y0 = as_ranges(y0)
        t_native = t.tolist() if hasattr(t, 'tolist') else t
        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0,
            float(delta), float(k), float(q), float(A), float(Omega),
            is_scale,
            seed,
        ]
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        self.save_filename = os.path.join(
            root, f"{self.__class__.__name__}_{dataset_config_hash}.pt"
        )
        if not reload and os.path.exists(self.save_filename):
            self.load()
            return

        self.t = torch.FloatTensor(t)
        self.input_length = input_length
        self.state_dim = 2
        self.state_names = [r'$x_0$', r'$x_1$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        np.random.seed(seed)
        y0_array = self.get_initial_value_array(y0, n_samples)
        param_arrays = self.get_param_arrays([delta, k, q, A, Omega], n_samples)

        self.y = [None] * n_samples
        for i in range(n_samples):
            sample_params = [param_array[i] for param_array in param_arrays]
            # Tolerances tighter than odeint's defaults and a raised step
            # limit: the corrugation term oscillates quickly at large
            # velocities, which otherwise triggers "excess work" warnings.
            self.y[i] = scipy_odeint(
                self._helmholtz_ode_func, y0_array[i], t, args=tuple(sample_params),
                rtol=1e-9, atol=1e-10, mxstep=20000,
            )

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)
        self.estimate_all_derivatives()
        self.save()

    @staticmethod
    def _helmholtz_ode_func(yv, tv, delta, k, q, A, Omega):
        x0, x1 = yv
        dx0 = x1
        dx1 = -delta * x1 - k * x0 - q * x0 ** 2 + A * np.cos(Omega * x0)
        return [dx0, dx1]

    def get_params(self):
        return self.params

    @classmethod
    def get_standard_dataset(
        cls,
        root,
        n_samples,
        times,
        granularity,
        sampling_scheme="regular",
        input_length_factor=3,
        extrapolation_ratio=0.3,
        reload=False,
        cfg=None,
    ):
        """Build the train / id-test / ood-t2 / ood-t3 splits with
        extrapolation pairs."""
        T = times
        nT = granularity * T
        extra_T = T * (1 + extrapolation_ratio)
        extra_nT = int(nT * (1 + extrapolation_ratio))

        np.random.seed(42)

        if sampling_scheme == "irregular":
            base_t = np.linspace(0, int(T), int(nT))
            t_interp = np.sort(base_t + np.random.uniform(-0.1, 0.1, int(nT)))
            base_t_extra = np.linspace(0, int(extra_T), int(extra_nT))
            t_extrap = np.sort(base_t_extra + np.random.uniform(-0.1, 0.1, int(extra_nT)))
        else:
            t_interp = np.linspace(0, int(T), int(nT))
            t_extrap = np.linspace(0, int(extra_T), int(extra_nT))

        forecaster = cfg.dataset.forecaster

        # ID: parameters and initial-condition box from the config.
        y0_id = as_ranges(forecaster.y0_range)
        params_id = {name: float(forecaster[name])
                     for name in ('delta', 'k', 'q', 'A', 'Omega')}

        # OOD T2: larger-amplitude initial conditions (box scaled outward
        # from the origin, still inside the escape barrier), same parameters.
        t2_scale = float(forecaster.ood_t2_y0_scale)
        y0_ood_t2 = [(low * t2_scale, high * t2_scale) for low, high in y0_id]
        params_ood_t2 = dict(params_id)

        # OOD T3: same initial conditions, core parameters (k, q) scaled. The
        # barrier location -k/q is unchanged; only the well gets deeper.
        t3_scale = float(forecaster.ood_t3_core_scale)
        y0_ood_t3 = y0_id
        params_ood_t3 = dict(params_id, k=params_id['k'] * t3_scale,
                             q=params_id['q'] * t3_scale)

        def _build(t_grid, y0, params):
            return cls(
                n_samples=n_samples, t=t_grid, y0=y0, **params,
                root=root, reload=reload,
            )

        train_data = _build(t_interp, y0_id, params_id)
        train_data_extra = _build(t_extrap, y0_id, params_id)
        id_test_data = _build(t_interp, y0_id, params_id)
        id_test_data_extra = _build(t_extrap, y0_id, params_id)
        ood_test_data_t2 = _build(t_interp, y0_ood_t2, params_ood_t2)
        ood_test_data_t2_extra = _build(t_extrap, y0_ood_t2, params_ood_t2)
        ood_test_data_t3 = _build(t_interp, y0_ood_t3, params_ood_t3)
        ood_test_data_t3_extra = _build(t_extrap, y0_ood_t3, params_ood_t3)

        return [[
            (train_data, train_data_extra),
            (id_test_data, id_test_data_extra),
            (ood_test_data_t2, ood_test_data_t2_extra),
            (ood_test_data_t3, ood_test_data_t3_extra),
        ]]

    def compute_true_W(self, feature_library, train_data):
        """Ground-truth coefficients in the library (4 in-library terms):
            x0' = x1
            x1' = -delta*x1 - k*x0 - q*x0^2
        The corrugation A*cos(Omega*x0) is out of the library and absent here.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()
        device = train_data.y.device

        W_true_dict = {f: torch.zeros(2, dtype=torch.float32, device=device)
                       for f in feature_names}

        if 'x1' in W_true_dict:
            W_true_dict['x1'][0] = 1.0
            W_true_dict['x1'][1] = -self.params['delta']
        if 'x0' in W_true_dict:
            W_true_dict['x0'][1] = -self.params['k']
        if 'x0^2' in W_true_dict:
            W_true_dict['x0^2'][1] = -self.params['q']

        self.W_true = W_true_dict
        return W_true_dict
