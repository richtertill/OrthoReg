"""SIR epidemic model with time-modulated memory terms (paper App. F.3).

The dynamics add one modulated term per compartment to the normalized SIR
model, with w_S = w_fast*(1 + S), w_I = w_fast*(1 + 2I), w_R = w_slow*(1 + R):

    S' = -beta*S*I/(S+I+R)
         + eps1 * S*I * sin(w_S*t) * exp(-t/tau) * tanh(w_slow*(I + R))
    I' =  beta*S*I/(S+I+R) - gamma*I
         + eps2 * I*R * cos(w_I*t) * exp(-t/tau) * sin(w_fast*S*I/(S + I + c_reg))
    R' =  gamma*I
         + eps3 * R*S * sin(w_R*t) * exp(-t/tau) * cos(w_slow*S*R/(R + c_reg))

The modulated terms are not expressible in the library; they are the missing
dynamics the neural component is asked to capture. eps_i -> 0 recovers the
classical normalized SIR model.
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


class SIREpidemicDataset(SeriesDataset):
    """SIR epidemic model with out-of-library memory-modulated terms.

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
        Per-state initial-condition ranges (S, I, R).
    beta, gamma: float
        Infection and recovery rates.
    epsilon1, epsilon2, epsilon3: float
        Strength of the out-of-library term in each compartment.
    omega_fast, omega_slow: float
        Base frequencies of the state-dependent modulation.
    tau_memory: float
        Decay time of the exp(-t/tau) envelope.
    c_reg: float
        Small constant that keeps the ratios inside the modulation finite.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(0.8, 1.0), (0.0, 0.2), (0.0, 0.1)],
        beta=2.0,
        gamma=1.0,
        epsilon1=0.3,
        epsilon2=0.25,
        epsilon3=0.2,
        omega_fast=3.0,
        omega_slow=1.5,
        tau_memory=2.0,
        c_reg=1e-6,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({
            'beta': beta,
            'gamma': gamma,
            'epsilon1': epsilon1,
            'epsilon2': epsilon2,
            'epsilon3': epsilon3,
            'omega_fast': omega_fast,
            'omega_slow': omega_slow,
            'tau_memory': tau_memory,
            'c_reg': c_reg,
        })

        self.true_equation = {
            'x0': (f"S' = -{beta}*S*I/(S+I+R) + {epsilon1}*S*I*sin(w_S*t)*exp(-t/{tau_memory})"
                   f"*tanh({omega_slow}*(I+R))"),
            'x1': (f"I' = {beta}*S*I/(S+I+R) - {gamma}*I + {epsilon2}*I*R*cos(w_I*t)"
                   f"*exp(-t/{tau_memory})*sin({omega_fast}*S*I/(S+I+{c_reg}))"),
            'x2': (f"R' = {gamma}*I + {epsilon3}*R*S*sin(w_R*t)*exp(-t/{tau_memory})"
                   f"*cos({omega_slow}*S*R/(R+{c_reg}))"),
        }

        os.makedirs(root, exist_ok=True)
        y0 = as_ranges(y0)
        t_native = t.tolist() if hasattr(t, 'tolist') else t
        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0,
            float(beta), float(gamma),
            float(epsilon1), float(epsilon2), float(epsilon3),
            float(omega_fast), float(omega_slow), float(tau_memory),
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
        self.state_dim = 3
        self.state_names = [r'$S$', r'$I$', r'$R$']
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        y0_array = self.get_initial_value_array(y0, n_samples)
        param_arrays = self.get_param_arrays(
            [beta, gamma, epsilon1, epsilon2, epsilon3,
             omega_fast, omega_slow, tau_memory],
            n_samples,
        )

        self.y = [None] * n_samples
        for i in range(n_samples):
            sample_params = [param_array[i] for param_array in param_arrays]
            self.y[i] = scipy_odeint(
                self._sir_ode_func, y0_array[i], t, args=(*sample_params, c_reg),
            )

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)
        self.estimate_all_derivatives()
        self.save()

    @staticmethod
    def _sir_ode_func(yv, tv, beta, gamma, epsilon1, epsilon2, epsilon3,
                      omega_fast, omega_slow, tau_memory, c_reg):
        s, i, r = yv

        # Normalized SIR backbone.
        incidence = beta * s * i / (s + i + r)
        standard_s = -incidence
        standard_i = incidence - gamma * i
        standard_r = gamma * i

        # Exponentially decaying envelope and state-dependent frequencies.
        memory_kernel = np.exp(-tv / tau_memory)
        omega_s = omega_fast * (1 + s)
        omega_i = omega_fast * (1 + 2 * i)
        omega_r = omega_slow * (1 + r)

        # Out-of-library modulated terms, one per compartment.
        modulated_s = (epsilon1 * s * i * np.sin(omega_s * tv) * memory_kernel
                       * np.tanh(omega_slow * (i + r)))
        modulated_i = (epsilon2 * i * r * np.cos(omega_i * tv) * memory_kernel
                       * np.sin(omega_fast * s * i / (s + i + c_reg)))
        modulated_r = (epsilon3 * r * s * np.sin(omega_r * tv) * memory_kernel
                       * np.cos(omega_slow * s * r / (r + c_reg)))

        return [standard_s + modulated_s, standard_i + modulated_i, standard_r + modulated_r]

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
        param_names = ('beta', 'gamma', 'epsilon1', 'epsilon2', 'epsilon3',
                       'omega_fast', 'omega_slow', 'tau_memory')

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
        """Ground-truth coefficients in the library (4 in-library terms).

        The incidence beta*S*I/(S+I+R) is represented by its normalized-
        population form beta*S*I (exact when S+I+R = 1):
            S' = -beta*S*I
            I' =  beta*S*I - gamma*I
            R' =  gamma*I
        The modulated memory terms are out of the library and absent here.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()
        device = train_data.y.device

        W_true_dict = {f: torch.zeros(3, dtype=torch.float32, device=device)
                       for f in feature_names}

        if 'x0 x1' in W_true_dict:
            W_true_dict['x0 x1'][0] = -self.params['beta']
            W_true_dict['x0 x1'][1] = self.params['beta']
        if 'x1' in W_true_dict:
            W_true_dict['x1'][1] = -self.params['gamma']
            W_true_dict['x1'][2] = self.params['gamma']

        self.W_true = W_true_dict
        return W_true_dict
