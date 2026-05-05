"""Modified damped pendulum dataset (paper Section 5).

The dynamics

    x0' = x1
    x1' = -omega0**2 * sin(x0) - alpha * x1
          + beta1 * cos(3*x0) + beta2 * exp(-x0**2) + beta3 * tanh(x1)

are designed so the polynomial library cannot express the cos / exp / tanh
terms. The hybrid model must capture them with the neural augmentation, which
makes this a clean test of orthogonal vs. L2 regularization.
"""

import json
import os
from collections import OrderedDict
from hashlib import sha1

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.integrate import odeint as scipy_odeint

from orthoreg.data.datasets.utils import SeriesDataset
from orthoreg.paths import DATA_DIR


def _eval_numeric_expr(expr):
    """Resolve simple Python / pi expressions written in YAML."""
    if isinstance(expr, str):
        expr = expr.replace("np.pi", "3.141592653589793")
        expr = expr.replace("pi", "3.141592653589793")
        try:
            return eval(expr)
        except Exception:
            return float(expr)
    return expr


class PendulumDataset(SeriesDataset):
    """Modified damped pendulum (paper Section 5)."""

    def __init__(
        self,
        n_samples,
        t,
        input_length=1,
        y0=[(-2, 2), (-2, 2)],
        omega0=1.0,
        alpha=0.2,
        beta1=0.3,
        beta2=0.25,
        beta3=0.15,
        is_scale=False,
        max_for_scaling=None,
        seed=0,
        root=DATA_DIR,
        reload=False,
        noise_level=0.0,
    ):
        super().__init__(max_for_scaling=max_for_scaling)

        self.params = OrderedDict({
            "omega0": omega0,
            "alpha": alpha,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
        })

        os.makedirs(root, exist_ok=True)
        dataset_config = OmegaConf.to_container(
            OmegaConf.create({
                "n_samples": n_samples,
                "t": t.tolist() if hasattr(t, "tolist") else t,
                "input_length": input_length,
                "y0": y0,
                "omega0": omega0,
                "alpha": alpha,
                "beta1": beta1,
                "beta2": beta2,
                "beta3": beta3,
                "is_scale": is_scale,
                "seed": seed,
                "noise_level": noise_level,
            }),
            resolve=True,
        )

        self.true_equation = {
            "x0": "x0' = x1",
            "x1": (
                f"x1' = -{omega0}^2*sin(x0) - {alpha}*x1"
                f" + {beta1}*cos(3*x0) + {beta2}*exp(-x0^2) + {beta3}*tanh(x1)"
            ),
        }

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
        self.state_names = [r"$x_0$", r"$x_1$"]
        if len(y0) != self.state_dim:
            raise AttributeError(
                f"Dimension of initial value y0 should be {self.state_dim}"
            )

        def _pendulum_ode_func(y, t, omega0, alpha, beta1, beta2, beta3):
            x0, x1 = y
            return [
                x1,
                -omega0 ** 2 * np.sin(x0)
                - alpha * x1
                + beta1 * np.cos(3 * x0)
                + beta2 * np.exp(-x0 ** 2)
                + beta3 * np.tanh(x1),
            ]

        np.random.seed(seed)
        y0_array = self.get_initial_value_array(y0, n_samples)

        param_arrays = self.get_param_arrays(
            [omega0, alpha, beta1, beta2, beta3], n_samples
        )
        omega0_array, alpha_array, beta1_array, beta2_array, beta3_array = param_arrays

        self.y = [None] * n_samples
        for i in range(n_samples):
            self.y[i] = scipy_odeint(
                _pendulum_ode_func,
                y0_array[i],
                t,
                args=(
                    omega0_array[i],
                    alpha_array[i],
                    beta1_array[i],
                    beta2_array[i],
                    beta3_array[i],
                ),
            )

        self.y = torch.FloatTensor(np.stack(self.y))

        if noise_level > 0:
            np.random.seed(seed)
            noise = np.random.normal(0, noise_level, size=self.y.shape)
            self.y = self.y + torch.FloatTensor(noise)

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
        sampling_scheme="irregular",
        input_length_factor=3,
        extrapolation_ratio=0.3,
        reload=False,
        cfg=None,
    ):
        T = times
        nT = granularity * T
        extra_T = T * (1 + extrapolation_ratio)
        extra_nT = int(nT * (1 + extrapolation_ratio))

        np.random.seed(42)

        if sampling_scheme == "irregular":
            base_t = np.linspace(0, int(T), int(nT))
            t_interp = np.sort(base_t + np.random.uniform(-0.1, 0.1, int(nT)))
            base_t_extra = np.linspace(0, int(extra_T), int(extra_nT))
            t_extrap = np.sort(
                base_t_extra + np.random.uniform(-0.1, 0.1, int(extra_nT))
            )
        else:
            t_interp = np.linspace(0, int(T), int(nT))
            t_extrap = np.linspace(0, int(extra_T), int(extra_nT))

        # Read ID parameters from the Hydra config. y0_range may contain
        # string expressions like "-pi/2"; resolve them here once.
        y0_id = cfg.dataset.forecaster.y0_range
        if isinstance(y0_id, str):
            import ast

            y0_id = ast.literal_eval(y0_id)
        y0_id = [
            [_eval_numeric_expr(val) for val in sublist] for sublist in y0_id
        ]

        omega0_id = float(cfg.dataset.forecaster.omega0)
        alpha_id = float(cfg.dataset.forecaster.alpha)
        beta1_id = float(cfg.dataset.forecaster.beta1)
        beta2_id = float(cfg.dataset.forecaster.beta2)
        beta3_id = float(cfg.dataset.forecaster.beta3)
        noise_level = (
            cfg.dataset.forecaster.get("noise_level", 0.0)
            if cfg and hasattr(cfg.dataset, "forecaster")
            else 0.0
        )

        # OOD T2: shifted initial conditions, same parameters.
        y0_ood_t2 = [(-5.0, 5.0), (-5.0, 5.0)]
        # OOD T3: same initial conditions, parameters scaled by 1.2.
        y0_ood_t3 = y0_id
        omega0_ood_t3 = omega0_id * 1.2
        alpha_ood_t3 = alpha_id * 1.2
        beta1_ood_t3 = beta1_id * 1.2
        beta2_ood_t3 = beta2_id * 1.2
        beta3_ood_t3 = beta3_id * 1.2

        def _build(t_grid, y0, omega0, alpha, beta1, beta2, beta3):
            return cls(
                n_samples=n_samples, t=t_grid, y0=y0,
                omega0=omega0, alpha=alpha,
                beta1=beta1, beta2=beta2, beta3=beta3,
                noise_level=noise_level,
            )

        train_data = _build(t_interp, y0_id, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        train_data_extra = _build(t_extrap, y0_id, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        id_test_data = _build(t_interp, y0_id, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        id_test_data_extra = _build(t_extrap, y0_id, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        ood_test_data_t2 = _build(t_interp, y0_ood_t2, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        ood_test_data_t2_extra = _build(t_extrap, y0_ood_t2, omega0_id, alpha_id, beta1_id, beta2_id, beta3_id)
        ood_test_data_t3 = _build(t_interp, y0_ood_t3, omega0_ood_t3, alpha_ood_t3, beta1_ood_t3, beta2_ood_t3, beta3_ood_t3)
        ood_test_data_t3_extra = _build(t_extrap, y0_ood_t3, omega0_ood_t3, alpha_ood_t3, beta1_ood_t3, beta2_ood_t3, beta3_ood_t3)

        return [[
            (train_data, train_data_extra),
            (id_test_data, id_test_data_extra),
            (ood_test_data_t2, ood_test_data_t2_extra),
            (ood_test_data_t3, ood_test_data_t3_extra),
        ]]

    def compute_true_W(self, feature_library, train_data):
        """Build the ground-truth coefficient dictionary in the polynomial
        library. The cos / exp / tanh terms are intentionally absent: they
        are the missing dynamics the neural augmentation must capture.
        """
        feature_library.fit(train_data.y, train_data.t)
        feature_names = feature_library.get_feature_names()

        W_true_dict = {
            feature: torch.zeros(2, dtype=torch.float32, device=train_data.y.device)
            for feature in feature_names
        }

        if "x1" in W_true_dict:
            W_true_dict["x1"][0] = 1.0  # x0' = x1

        omega0 = self.params["omega0"]
        alpha = self.params["alpha"]

        if "sin(1 x0)" in W_true_dict:
            W_true_dict["sin(1 x0)"][1] = -omega0 ** 2
        if "x1" in W_true_dict:
            W_true_dict["x1"][1] = -alpha

        self.W_true = W_true_dict
        return W_true_dict
