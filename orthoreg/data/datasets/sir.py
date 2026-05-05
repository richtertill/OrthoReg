from collections import OrderedDict
from hashlib import sha1
import json
import os

import numpy as np
import torch
from scipy.integrate import odeint as scipy_odeint

from orthoreg.data.datasets.utils import SeriesDataset
from orthoreg.paths import DATA_DIR


class SIREpidemicDataset(SeriesDataset):
    """SIR epidemic model used in the paper.

        dS/dt = -beta * I * S / (S + I + R)
        dI/dt =  beta * I * S / (S + I + R) - gamma * I
        dR/dt =  gamma * I

    The true dynamics contain a rational nonlinearity that is *not*
    expressible in the polynomial library; it is the missing dynamics
    that the neural component is asked to capture.
    """

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(90, 100), (0, 5), (0, 0)],
        beta=4,
        gamma=0.4,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({'beta': beta, 'gamma': gamma})

        os.makedirs(root, exist_ok=True)
        t_native = t.tolist() if hasattr(t, 'tolist') else t
        dataset_config = [
            n_samples,
            t_native,
            input_length,
            y0,
            beta,
            gamma,
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

        def sir_ode_func(yv, tv, beta, gamma):
            s, i_state, r = yv
            denom = s + i_state + r
            return (
                -beta * i_state * s / denom,
                beta * i_state * s / denom - gamma * i_state,
                gamma * i_state,
            )

        y0_array = self.get_initial_value_array(y0, n_samples)
        param_arrays = self.get_param_arrays([beta, gamma], n_samples)
        beta_array, gamma_array = param_arrays

        self.y = [None] * n_samples
        for i in range(n_samples):
            args = (beta_array[i], gamma_array[i])
            self.y[i] = scipy_odeint(sir_ode_func, y0_array[i], t, args=args)

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
        input_length_factor=5,
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

        y0_id = [(9, 10), (1, 5), (0, 0)]
        beta_id, gamma_id = 4, 0.4

        y0_ood_t2 = [(90, 100), (1, 5), (0, 0)]
        beta_ood_t2, gamma_ood_t2 = beta_id, gamma_id

        y0_ood_t3 = [(90, 100), (1, 5), (0, 0)]
        beta_ood_t3 = (8, 12)
        gamma_ood_t3 = (0.8, 1.2)

        n_train = int(n_samples * 0.8)
        n_test = max(1, int(n_samples * 0.2))

        def _build(t_grid, n, y0, beta, gamma, max_scale=None):
            return cls(
                n, t_grid,
                input_length=input_length,
                y0=y0, beta=beta, gamma=gamma,
                seed=0,
                is_scale=is_scale,
                max_for_scaling=max_scale,
                root=root,
                reload=reload,
            )

        train_data = _build(t_interp, n_train, y0_id, beta_id, gamma_id)
        train_data_extra = _build(t_extrap, n_train, y0_id, beta_id, gamma_id,
                                  max_scale=train_data.max_for_scaling)

        id_test = _build(t_interp, n_test, y0_id, beta_id, gamma_id,
                         max_scale=train_data.max_for_scaling)
        id_test_extra = _build(t_extrap, n_test, y0_id, beta_id, gamma_id,
                               max_scale=train_data.max_for_scaling)

        ood_t2 = _build(t_interp, n_test, y0_ood_t2, beta_ood_t2, gamma_ood_t2,
                        max_scale=train_data.max_for_scaling)
        ood_t2_extra = _build(t_extrap, n_test, y0_ood_t2, beta_ood_t2, gamma_ood_t2,
                              max_scale=train_data.max_for_scaling)

        ood_t3 = _build(t_interp, n_test, y0_ood_t3, beta_ood_t3, gamma_ood_t3,
                        max_scale=train_data.max_for_scaling)
        ood_t3_extra = _build(t_extrap, n_test, y0_ood_t3, beta_ood_t3, gamma_ood_t3,
                              max_scale=train_data.max_for_scaling)

        return [[
            (train_data, train_data_extra),
            (id_test, id_test_extra),
            (ood_t2, ood_t2_extra),
            (ood_t3, ood_t3_extra),
        ]]

    def compute_true_W(self, feature_library, train_data):
        """Approximate ground-truth coefficients in the polynomial library.

        SIR has a rational nonlinearity beta*I*S/(S+I+R) that is not
        expressible polynomially. We project it onto the bilinear x0*x1
        interaction term as the closest available match - the cubic-and-
        higher reminder is exactly the missing dynamics OrthoReg should
        leave to the neural component.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()
        device = train_data.y.device

        W_true_dict = {f: torch.zeros(3, dtype=torch.float32, device=device)
                       for f in feature_names}

        for feature in feature_names:
            if feature in ("x0 x1", "x0 * x1"):
                W_true_dict[feature][0] = torch.tensor([-self.params['beta']],
                                                       dtype=torch.float32, device=device)
                W_true_dict[feature][1] = torch.tensor([self.params['beta']],
                                                       dtype=torch.float32, device=device)
            elif feature == "x1":
                W_true_dict[feature][1] = torch.tensor([-self.params['gamma']],
                                                       dtype=torch.float32, device=device)
                W_true_dict[feature][2] = torch.tensor([self.params['gamma']],
                                                       dtype=torch.float32, device=device)

        self.W_true = W_true_dict
        return W_true_dict
