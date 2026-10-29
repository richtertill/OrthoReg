"""Lotka-Volterra predator-prey system with modulated coupling (paper App. F.1).

The dynamics

    x0' = alpha*x0 - beta*x0*x1
          + eps1 * x0 * sin(w_fast*t) * cos(w_fast*x0*x1) * sin(w_slow*(x0 + x1))
    x1' = delta*x0*x1 - gamma*x1
          + eps2 * x1 * sin(w_fast*t) * cos(w_fast*x0*x1) * sin(w_slow*(x0 + x1))
                     * sin(x0 / (x1 + c_reg))

combine the classical predator-prey terms (all in the polynomial library)
with a high-frequency, state-modulated coupling that the library cannot
express. The explicit sin(w_fast*t) factor makes the system non-autonomous.
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


class LotkaVolterraDataset(SeriesDataset):
    """Lotka-Volterra system with out-of-library modulated coupling.

    The standard split provides
        train, id_test, ood_t2, ood_t3,
    where OOD-T2 widens the initial-condition box and OOD-T3 scales every
    parameter by ``ood_t3_param_scale``. Each split is also exposed on a
    longer extrapolation time grid (``*_extra``).

    Parameters
    ----------
    n_samples: int
        Number of trajectories.
    t: np.ndarray
        Time points used to integrate the ODE.
    y0: list
        Per-state initial-condition ranges (prey, predator).
    alpha, beta, gamma, delta: float
        Classical Lotka-Volterra rates.
    epsilon1, epsilon2: float
        Strength of the out-of-library coupling in each equation.
    omega_fast, omega_slow: float
        Frequencies of the fast modulation and the slow envelope.
    c_reg: float
        Small constant that keeps x0 / (x1 + c_reg) finite.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(0.5, 2.0), (0.5, 2.0)],
        alpha=5.0,
        beta=2.5,
        gamma=5.0,
        delta=2.5,
        epsilon1=3.0,
        epsilon2=2.0,
        omega_fast=10.0,
        omega_slow=2.0,
        c_reg=1e-6,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'epsilon1': epsilon1,
            'epsilon2': epsilon2,
            'omega_fast': omega_fast,
            'omega_slow': omega_slow,
            'c_reg': c_reg,
        })

        self.true_equation = {
            'x0': (f"x0' = {alpha}*x0 - {beta}*x0*x1 + {epsilon1}*x0*sin({omega_fast}*t)"
                   f"*cos({omega_fast}*x0*x1)*sin({omega_slow}*(x0+x1))"),
            'x1': (f"x1' = {delta}*x0*x1 - {gamma}*x1 + {epsilon2}*x1*sin({omega_fast}*t)"
                   f"*cos({omega_fast}*x0*x1)*sin({omega_slow}*(x0+x1))*sin(x0/(x1+{c_reg}))"),
        }

        os.makedirs(root, exist_ok=True)
        y0 = as_ranges(y0)
        t_native = t.tolist() if hasattr(t, 'tolist') else t
        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0,
            float(alpha), float(beta), float(gamma), float(delta),
            float(epsilon1), float(epsilon2), float(omega_fast), float(omega_slow),
            float(c_reg),
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
        self.state_names = [r'$x$', r'$y$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        y0_array = self.get_initial_value_array(y0, n_samples)
        param_arrays = self.get_param_arrays(
            [alpha, beta, gamma, delta, epsilon1, epsilon2, omega_fast, omega_slow],
            n_samples,
        )

        self.y = [None] * n_samples
        for i in range(n_samples):
            sample_params = [param_array[i] for param_array in param_arrays]
            self.y[i] = scipy_odeint(
                self._lotka_volterra_ode_func, y0_array[i], t,
                args=(*sample_params, c_reg),
            )

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)
        self.estimate_all_derivatives()
        self.save()

    @staticmethod
    def _lotka_volterra_ode_func(yv, tv, alpha, beta, gamma, delta,
                                 epsilon1, epsilon2, omega_fast, omega_slow, c_reg):
        prey, predator = yv

        # Classical predator-prey terms (in the library).
        standard_prey = alpha * prey - beta * prey * predator
        standard_predator = delta * prey * predator - gamma * predator

        # Fast time modulation of the coupling times a slow state envelope
        # (out of the library).
        fast_modulation = np.sin(omega_fast * tv) * np.cos(omega_fast * prey * predator)
        slow_envelope = np.sin(omega_slow * (prey + predator))

        coupling_prey = epsilon1 * prey * fast_modulation * slow_envelope
        coupling_predator = (epsilon2 * predator * fast_modulation * slow_envelope
                             * np.sin(prey / (predator + c_reg)))

        return [standard_prey + coupling_prey, standard_predator + coupling_predator]

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
        param_names = ('alpha', 'beta', 'gamma', 'delta',
                       'epsilon1', 'epsilon2', 'omega_fast', 'omega_slow')

        # ID: parameters and initial-condition box from the config.
        y0_id = as_ranges(forecaster.y0_range)
        params_id = {name: float(forecaster[name]) for name in param_names}

        # OOD T2: wider initial-condition box, same parameters.
        y0_ood_t2 = as_ranges(forecaster.ood_t2_y0_range)
        params_ood_t2 = dict(params_id)

        # OOD T3: same initial conditions, every parameter scaled.
        y0_ood_t3 = y0_id
        scale = float(forecaster.ood_t3_param_scale)
        params_ood_t3 = {name: value * scale for name, value in params_id.items()}

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
            x0' = alpha*x0 - beta*x0*x1
            x1' = delta*x0*x1 - gamma*x1
        The modulated coupling terms are out of the library and absent here.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()
        device = train_data.y.device

        W_true_dict = {f: torch.zeros(2, dtype=torch.float32, device=device)
                       for f in feature_names}

        if 'x0' in W_true_dict:
            W_true_dict['x0'][0] = self.params['alpha']
        if 'x0 x1' in W_true_dict:
            W_true_dict['x0 x1'][0] = -self.params['beta']
            W_true_dict['x0 x1'][1] = self.params['delta']
        if 'x1' in W_true_dict:
            W_true_dict['x1'][1] = -self.params['gamma']

        self.W_true = W_true_dict
        return W_true_dict
