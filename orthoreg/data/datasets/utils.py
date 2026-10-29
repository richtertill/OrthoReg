"""Base time-series dataset used by the paper systems."""

import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import torch
from torch.utils.data import Dataset

from orthoreg.paths import RESULT_DIR


class SeriesDataset(ABC, Dataset):
    """
    Abstract class for Time Series Datasets
    y, t
    """

    def __init__(self, max_for_scaling=None):
        # y shape: (n_samples, time_steps, dimension)
        # t shape: (time_steps)

        self.state_dim = None
        self.state_names = None
        self.y = None
        self.dy = None      # Estimated derivatives
        self.t = None
        self.input_length = None
        self.max_for_scaling = max_for_scaling
        self.phy_params = None      # Fitted parameters

    def plot(self, dim=0, **kwargs):
        unscaled_y = self.return_unscaled_y()
        for i in range(len(self)):
            plt.plot(
                self.t.numpy(),
                unscaled_y[i, :, dim].numpy(),
                label=f"y(t): dimension {dim}",
            )
        if self.input_length > 0:
            plt.axvline(
                x=self.t.numpy()[self.input_length - 1], linestyle="--", color="black"
            )

        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if "xlim" in kwargs:
            plt.xlim(kwargs["xlim"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            plt.ylabel(kwargs["ylabel"])
        if "title" in kwargs:
            plt.title(kwargs["title"])
        os.makedirs(RESULT_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULT_DIR, f"{kwargs['title']}_dim_{dim}.png"))
        plt.show()

    def scale(self, is_scale=False):
        if self.max_for_scaling is None:
            if is_scale:
                self.max_for_scaling = self.y.amax(dim=[0, 1]) / 10.
            else:
                self.max_for_scaling = torch.ones(self.state_dim)

        self.y = self.y / self.max_for_scaling

    def return_unscaled_y(self):
        return self.y * self.max_for_scaling


    def estimate_derivatives(self, method="smooth"):
        if self.y is None:
            return

        t = self.t.numpy()
        if method == "smooth":
            differentiation_method = ps.SmoothedFiniteDifference(
                order=2, smoother_kws={"window_length": 5}
            )
        else:
            differentiation_method = ps.FiniteDifference(order=2)

        dy = []
        for i in range(self.y.shape[0]):
            y_i = self.y[i].numpy()
            dy.append(differentiation_method._differentiate(y_i, t))
        return torch.tensor(np.stack(dy))

    def estimate_all_derivatives(self):
        self.dy = {"smooth": self.estimate_derivatives(method="smooth")}
    
    def get_initial_value_array(self, y0, n_samples):
        # Each entry of y0 is either a fixed value or a (low, high) range that
        # is sampled uniformly. Ranges may come as tuples, lists or Hydra
        # ListConfigs; all of them are treated as ranges.
        initial_value_array = []
        for i in range(self.state_dim):
            if np.isscalar(y0[i]):
                array = np.tile(float(y0[i]), n_samples)
            else:
                low, high = (float(bound) for bound in y0[i])
                array = np.random.uniform(low, high, n_samples)
            initial_value_array.append(array)

        initial_value_array = np.stack(initial_value_array, axis=1)
        return initial_value_array

    def get_param_arrays(self, params, n_samples):
        param_arrays = []
        for param in params:
            if isinstance(param, tuple):
                param_array = np.random.uniform(*param, n_samples)
            else:
                param_array = np.tile(param, n_samples)
            param_arrays.append(param_array)

        return param_arrays

    def save(self):
        with open(self.save_filename, "wb") as f:
            all_var = [
                self.state_names,
                self.state_dim,
                self.input_length,
                self.t,
                self.y,
                self.dy,
                self.max_for_scaling,
                self.phy_params,
            ]
            torch.save(all_var, f)

    def load(self):
        print(f"Using saved file: {self.save_filename}")
        (
            self.state_names,
            self.state_dim,
            self.input_length,
            self.t,
            self.y,
            self.dy,
            self.max_for_scaling,
            self.phy_params,
        ) = torch.load(self.save_filename)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return idx, self.y[idx]


def as_ranges(y0):
    """Convert initial-condition specs to plain Python types.

    Ranges become ``(low, high)`` tuples and fixed values floats, so the spec
    can be hashed (dataset cache key) and sampled consistently.
    """
    return [
        float(entry) if np.isscalar(entry) else tuple(float(b) for b in entry)
        for entry in y0
    ]
