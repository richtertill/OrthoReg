from orthoreg.data.datasets.utils import SeriesDataset
from scipy.integrate import odeint as scipy_odeint
import numpy as np
import torch
import os
from hashlib import sha1
import json
from collections import OrderedDict
from omegaconf import OmegaConf
from orthoreg.paths import DATA_DIR


class DuffingOscillatorDataset(SeriesDataset):
    """
    Generate unforced Duffing oscillator data following Goring et al. (2024).

    The unforced Duffing oscillator (Goring et al., 2024, Appendix D.1):
        dx/dt = y
        dy/dt = a*y - x*(b + c*x^2)

    For [a, b, c] = [-1/2, -1, 1/10] the system is multistable, with two
    coexisting stable equilibria. Training trajectories are sampled from the
    positive basin only; the OOD splits probe cross-basin generalization
    (T2) and parameter shifts (T3).

    Parameters
    ----------
    n_samples: int
        Number of trajectories.
    t: np.ndarray
        Time points used to integrate the ODE.
    input_length: int
        Length of the input sequence (kept for API compatibility).
    y0: list
        Per-state initial-condition ranges.
    a, b, c: float
        Duffing parameters (damping, linear stiffness, cubic stiffness).
    is_scale: bool
        Whether to scale the data.
    max_for_scaling: float, optional
        Maximum value for scaling.
    seed: int
        Random seed.
    root: str
        Root directory used to cache the generated tensors.
    reload: bool
        If True, regenerate even when a cached file exists.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(-2, 2), (-2, 2)],
        a=-0.5,
        b=-1.0,
        c=0.1,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({'a': a, 'b': b, 'c': c})

        self.true_equation = {
            'x': 'y',
            'y': f'{a}*y - x*({b} + {c}*x^2)',
        }

        os.makedirs(root, exist_ok=True)

        # Convert any potential OmegaConf objects to native Python types so
        # the config hash is deterministic across runs.
        y0_native = [list(r) if hasattr(r, '_content') else r for r in y0]
        t_native = t.tolist() if hasattr(t, 'tolist') else t

        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0_native,
            float(a),
            float(b),
            float(c),
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

        # Sample initial values and parameters from the given ranges.
        y0_array = self.get_initial_value_array(y0, n_samples)
        param_arrays = self.get_param_arrays([a, b, c], n_samples)
        a_array, b_array, c_array = param_arrays

        self.y = [None] * n_samples
        for i in range(n_samples):
            self.y[i] = scipy_odeint(
                lambda yv, tv: self._duffing_ode_func(
                    yv, tv, a_array[i], b_array[i], c_array[i]
                ),
                y0_array[i],
                t,
            )

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)
        self.estimate_all_derivatives()
        self.save()

    def _duffing_ode_func(self, y, t, a, b, c):
        """Unforced Duffing oscillator ODE."""
        x, y_v = y[0], y[1]
        dxdt = y_v
        dydt = a * y_v - x * (b + c * x ** 2)
        return [dxdt, dydt]

    def get_params(self):
        """Return the parameters as an OrderedDict."""
        return self.params

    @classmethod
    def get_standard_dataset(
        cls,
        root,
        n_samples,
        times,
        granularity,
        sampling_scheme="irregular",
        input_length_factor=3,
        extrapolation_ratio=0.3,
        reload=False,
        cfg=None,
    ):
        """Create the train/ID-test/OOD-T2/OOD-T3 splits used in the paper.

        Training and ID-test trajectories are sampled from the positive basin.
        OOD-T2 evaluates cross-basin generalization (negative basin, same
        parameters); OOD-T3 evaluates parameter shifts (negative basin,
        slightly different a, b, c).
        """
        T = times
        nT = granularity * T
        extra_T = T * (1 + extrapolation_ratio)
        extra_nT = int(nT * (1 + extrapolation_ratio))

        np.random.seed(42)

        if sampling_scheme == "irregular":
            base_t = np.linspace(0, int(T), int(nT))
            noise = np.random.uniform(-0.1, 0.1, int(nT))
            t_interp = base_t + noise
            t_interp.sort()

            base_t_extra = np.linspace(0, int(extra_T), int(extra_nT))
            noise_extra = np.random.uniform(-0.1, 0.1, int(extra_nT))
            t_extrap = base_t_extra + noise_extra
            t_extrap.sort()
        else:
            t_interp = np.linspace(0, int(T), int(nT))
            t_extrap = np.linspace(0, int(extra_T), int(extra_nT))

        a_id = cfg.dataset.forecaster.a
        b_id = cfg.dataset.forecaster.b
        c_id = cfg.dataset.forecaster.c

        # For unforced Duffing with b < 0, c > 0, equilibria sit at
        # x = +/- sqrt(-b/c), y = 0.
        if b_id < 0 and c_id > 0:
            well_position = float(np.sqrt(-b_id / c_id))
        else:
            well_position = float(np.sqrt(1.0 / 0.1))

        # ID: positive basin only.
        y0_id = [(0.1, well_position * 0.8), (-1.0, 1.0)]

        # OOD T2: negative basin, same parameters.
        y0_ood_t2 = [(-well_position * 0.8, -0.1), (-1.0, 1.0)]
        a_ood_t2, b_ood_t2, c_ood_t2 = a_id, b_id, c_id

        # OOD T3: negative basin, shifted parameters (cubic stiffness doubled).
        y0_ood_t3 = [(-well_position * 0.8, -0.1), (-1.0, 1.0)]
        a_ood_t3 = a_id * 1.2
        b_ood_t3 = b_id * 1.2
        c_ood_t3 = c_id * 2.0

        train_data = cls(n_samples=n_samples, t=t_interp, y0=y0_id,
                         a=a_id, b=b_id, c=c_id)
        train_data_extra = cls(n_samples=n_samples, t=t_extrap, y0=y0_id,
                               a=a_id, b=b_id, c=c_id)

        id_test_data = cls(n_samples=n_samples, t=t_interp, y0=y0_id,
                           a=a_id, b=b_id, c=c_id)
        id_test_data_extra = cls(n_samples=n_samples, t=t_extrap, y0=y0_id,
                                 a=a_id, b=b_id, c=c_id)

        ood_test_data_t2 = cls(n_samples=n_samples, t=t_interp, y0=y0_ood_t2,
                               a=a_ood_t2, b=b_ood_t2, c=c_ood_t2)
        ood_test_data_t2_extra = cls(n_samples=n_samples, t=t_extrap, y0=y0_ood_t2,
                                     a=a_ood_t2, b=b_ood_t2, c=c_ood_t2)

        ood_test_data_t3 = cls(n_samples=n_samples, t=t_interp, y0=y0_ood_t3,
                               a=a_ood_t3, b=b_ood_t3, c=c_ood_t3)
        ood_test_data_t3_extra = cls(n_samples=n_samples, t=t_extrap, y0=y0_ood_t3,
                                     a=a_ood_t3, b=b_ood_t3, c=c_ood_t3)

        return [[
            (train_data, train_data_extra),
            (id_test_data, id_test_data_extra),
            (ood_test_data_t2, ood_test_data_t2_extra),
            (ood_test_data_t3, ood_test_data_t3_extra),
        ]]

    def compute_true_W(self, feature_library, train_data):
        """Build the ground-truth coefficient dict for the symbolic library.

        For the unforced Duffing oscillator the true dynamics are
            x' = y
            y' = a*y - b*x - c*x^3.
        The cubic term x^3 is intentionally omitted from the symbolic
        library: it is the missing dynamics that the neural component must
        capture. This makes Duffing a clean test of whether OrthoReg keeps
        the symbolic and neural parts disjoint under partial library mismatch.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()

        W_true_dict = {
            feature: torch.zeros(2, dtype=torch.float32,
                                 device=train_data.y.device)
            for feature in feature_names
        }

        # Equation 1: x' = y, where x maps to x0 and y maps to x1.
        if 'x1' in W_true_dict:
            W_true_dict['x1'][0] = 1.0

        a = self.params['a']
        b = self.params['b']
        c = self.params['c']

        # Equation 2: y' = a*y - b*x (cubic term is intentionally absent).
        if 'x1' in W_true_dict:
            W_true_dict['x1'][1] = a
        if 'x0' in W_true_dict:
            W_true_dict['x0'][1] = -b

        # If the library happens to include x^3 we explicitly mark it as
        # missing from the known physics so downstream metrics stay honest.
        for cubic_key in ('x^3', 'x**3', 'x0^3', 'x0**3'):
            if cubic_key in W_true_dict:
                W_true_dict[cubic_key][1] = 0.0

        self.W_true = W_true_dict

        print('W_true for unforced Duffing oscillator:')
        print(W_true_dict)
        print(f"True equations: x' = {self.true_equation['x']},"
              f" y' = {self.true_equation['y']}")
        print(f"Missing from library (must be learned by neural component):"
              f" -{c}*x^3")

        return W_true_dict
