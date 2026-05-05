from collections import OrderedDict
from hashlib import sha1
import json
import os

import numpy as np
import torch
from scipy.integrate import odeint as scipy_odeint

from orthoreg.data.datasets.utils import SeriesDataset
from orthoreg.paths import DATA_DIR


class LotkaVolterraDataset(SeriesDataset):
    """Lotka-Volterra predator-prey system used in the paper.

        dx/dt = alpha * x  - beta  * x * y
        dy/dt = delta * x * y - gamma * y

    The standard split provides
        train, id_test, ood_t2, ood_t3,
    where OOD-T2 perturbs the initial conditions and OOD-T3 perturbs both
    the initial conditions and the parameters. Each split is also exposed
    on a longer extrapolation time grid (`*_extra`).
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(1000, 2000), (10, 20)],
        alpha=0.1 * 12,
        beta=0.005 * 12,
        gamma=0.04 * 12,
        delta=0.00004 * 12,
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
        })

        os.makedirs(root, exist_ok=True)
        t_native = t.tolist() if hasattr(t, 'tolist') else t
        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0,
            alpha,
            beta,
            gamma,
            delta,
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

        def lotka_volterra_ode_func(yv, tv, alpha, beta, gamma, delta):
            prey, predator = yv
            return (
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            )

        y0_array = self.get_initial_value_array(y0, n_samples)

        param_arrays = self.get_param_arrays([alpha, beta, gamma, delta], n_samples)
        alpha_array, beta_array, gamma_array, delta_array = param_arrays

        self.y = [None] * n_samples
        for i in range(n_samples):
            args = (alpha_array[i], beta_array[i], gamma_array[i], delta_array[i])
            self.y[i] = scipy_odeint(lotka_volterra_ode_func, y0_array[i], t, args=args)

        self.y = torch.FloatTensor(np.stack(self.y))
        self.scale(is_scale)
        self.estimate_all_derivatives()
        self.save()

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

        input_length = int(nT // input_length_factor)
        is_scale = True

        # Base settings (ID).
        y0_id = [(1000, 2000), (10, 20)]
        alpha_id = 0.1 * 12
        beta_id = 0.005 * 12
        gamma_id = 0.04 * 12
        delta_id = 0.00004 * 12

        # OOD T2: shifted initial conditions, same parameters.
        y0_ood_t2 = [(100, 200), (10, 20)]
        alpha_ood_t2, beta_ood_t2 = alpha_id, beta_id
        gamma_ood_t2, delta_ood_t2 = gamma_id, delta_id

        # OOD T3: shifted initial conditions and parameters.
        y0_ood_t3 = [(100, 200), (10, 20)]
        alpha_ood_t3 = (0.2 * 12, 0.3 * 12)
        beta_ood_t3 = (0.01 * 12, 0.015 * 12)
        gamma_ood_t3 = (0.08 * 12, 0.12 * 12)
        delta_ood_t3 = (0.00008 * 12, 0.00012 * 12)

        n_train = int(n_samples * 0.8)
        n_test = max(1, int(n_samples * 0.2))

        def _build(t_grid, n, y0, alpha, beta, gamma, delta, max_scale=None):
            return cls(
                n, t_grid,
                input_length=input_length,
                y0=y0,
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                seed=0,
                is_scale=is_scale,
                max_for_scaling=max_scale,
                root=root,
                reload=reload,
            )

        train_data = _build(t_interp, n_train, y0_id, alpha_id, beta_id, gamma_id, delta_id)
        train_data_extra = _build(t_extrap, n_train, y0_id, alpha_id, beta_id, gamma_id, delta_id,
                                  max_scale=train_data.max_for_scaling)

        id_test_data = _build(t_interp, n_test, y0_id, alpha_id, beta_id, gamma_id, delta_id,
                              max_scale=train_data.max_for_scaling)
        id_test_data_extra = _build(t_extrap, n_test, y0_id, alpha_id, beta_id, gamma_id, delta_id,
                                    max_scale=train_data.max_for_scaling)

        ood_t2 = _build(t_interp, n_test, y0_ood_t2, alpha_ood_t2, beta_ood_t2,
                        gamma_ood_t2, delta_ood_t2, max_scale=train_data.max_for_scaling)
        ood_t2_extra = _build(t_extrap, n_test, y0_ood_t2, alpha_ood_t2, beta_ood_t2,
                              gamma_ood_t2, delta_ood_t2, max_scale=train_data.max_for_scaling)

        ood_t3 = _build(t_interp, n_test, y0_ood_t3, alpha_ood_t3, beta_ood_t3,
                        gamma_ood_t3, delta_ood_t3, max_scale=train_data.max_for_scaling)
        ood_t3_extra = _build(t_extrap, n_test, y0_ood_t3, alpha_ood_t3, beta_ood_t3,
                              gamma_ood_t3, delta_ood_t3, max_scale=train_data.max_for_scaling)

        return [[
            (train_data, train_data_extra),
            (id_test_data, id_test_data_extra),
            (ood_t2, ood_t2_extra),
            (ood_t3, ood_t3_extra),
        ]]

    def compute_true_W(self, feature_library, train_data):
        """Ground-truth coefficients in the polynomial library:
            x0' = alpha*x0 - beta*x0*x1
            x1' = -gamma*x1 + delta*x0*x1
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()
        device = train_data.y.device

        W_true_dict = {f: torch.zeros(2, dtype=torch.float32, device=device)
                       for f in feature_names}

        for feature in feature_names:
            if feature in ("x0 x1", "x0 * x1"):
                W_true_dict[feature][0] = torch.tensor([-self.params['beta']],
                                                       dtype=torch.float32, device=device)
                W_true_dict[feature][1] = torch.tensor([self.params['delta']],
                                                       dtype=torch.float32, device=device)
            elif feature == "x0":
                W_true_dict[feature][0] = torch.tensor([self.params['alpha']],
                                                       dtype=torch.float32, device=device)
            elif feature == "x1":
                W_true_dict[feature][1] = torch.tensor([-self.params['gamma']],
                                                       dtype=torch.float32, device=device)

        self.W_true = W_true_dict
        return W_true_dict
