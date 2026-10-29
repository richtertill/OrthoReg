"""Canonical SINDy baseline: PySINDy sequential thresholded least squares.

STLSQ (Brunton et al., 2016) selects a sparse support by repeatedly solving
a least-squares problem and zeroing coefficients below a threshold, then
refits the surviving terms without shrinkage. It runs on the same symbolic
library and train / ID / OOD splits as the hybrid models and is scored by
:class:`orthoreg.models.experiment.HybridExperiment.test_step`, so every
metric (derivative and state MSE, F1, FPR, full-support F1) is computed
exactly as for Pure / L2 / OrthoReg.

Settings follow the paper's appendix: ``STLSQ(threshold=0.045,
normalize_columns=False)`` with second-order finite differences
(``FiniteDifference(order=2)``) as derivative targets. With
``training.retrain=true`` the support selected on the training split is
kept fixed and the active coefficients are refit by ordinary least squares
on the OOD-T3 extrapolation trajectories, mirroring the hybrids' second
stage.
"""

import copy

import numpy as np
import pysindy as ps
import torch

from orthoreg.models.experiment import HybridExperiment


def _finite_difference_targets(trajectories, t):
    """Stack states and second-order finite-difference derivatives."""
    differentiation_method = ps.FiniteDifference(order=2)
    states, derivatives = [], []
    for trajectory in trajectories:
        states.append(trajectory)
        derivatives.append(differentiation_method._differentiate(trajectory, t))
    return np.concatenate(states), np.concatenate(derivatives)


def fit_stlsq(trajectories, t, feature_library, threshold=0.045):
    """Fit STLSQ on a set of trajectories.

    Parameters
    ----------
    trajectories : np.ndarray, shape ``[n_trajectories, n_times, state_dim]``
    t : np.ndarray, shape ``[n_times]``
    feature_library : pysindy feature library (copied, not modified)
    threshold : float
        STLSQ hard threshold.

    Returns
    -------
    coef : np.ndarray, shape ``[n_features, state_dim]``
        Coefficients in the ``PhysicalModel.coef_`` layout.
    feature_names : list of str
    """
    model = ps.SINDy(
        feature_library=copy.deepcopy(feature_library),
        optimizer=ps.STLSQ(threshold=threshold, normalize_columns=False),
        differentiation_method=ps.FiniteDifference(order=2),
    )
    model.fit(
        [trajectory for trajectory in trajectories],
        t=[t for _ in range(len(trajectories))],
        multiple_trajectories=True,
    )
    return model.coefficients().T, model.get_feature_names()


def refit_on_support(coef, trajectories, t, feature_library):
    """Ordinary least squares on the active support of ``coef``.

    Each state equation is refit on its own active columns; inactive
    coefficients stay exactly zero.
    """
    states, derivatives = _finite_difference_targets(trajectories, t)
    features = feature_library.transform(states)

    refit = np.zeros_like(coef)
    for k in range(coef.shape[1]):
        active = coef[:, k] != 0
        if np.any(active):
            solution, *_ = np.linalg.lstsq(features[:, active], derivatives[:, k], rcond=None)
            refit[active, k] = solution
    return refit


class STLSQExperiment(HybridExperiment):
    """Pure-symbolic experiment whose coefficients come from STLSQ.

    No gradient training: :meth:`fit_stlsq` writes the STLSQ solution into
    ``net.model_phy.coef_`` and the inherited ``test_step`` scores it.
    """

    def fit_stlsq(self, train_data, feature_library):
        coef, feature_names = fit_stlsq(
            train_data.y.cpu().numpy(),
            train_data.t.cpu().numpy(),
            feature_library,
            threshold=self.cfg.model.stlsq_threshold,
        )
        if list(feature_names) != list(self.net.model_phy.feature_names):
            raise ValueError("STLSQ and the hybrid model use different library features")
        self._set_coefficients(coef)
        return coef

    def refit_on_support(self, data, feature_library):
        coef = self.net.model_phy.coef_.detach().cpu().numpy()
        refit = refit_on_support(
            coef, data.y.cpu().numpy(), data.t.cpu().numpy(), feature_library,
        )
        self._set_coefficients(refit)
        return refit

    def _set_coefficients(self, coef):
        with torch.no_grad():
            self.net.model_phy.coef_.copy_(
                torch.as_tensor(coef, dtype=self.net.model_phy.coef_.dtype)
            )


def run_stlsq(cfg, feature_library, data_dict, datamodule, trainer, net):
    """Fit STLSQ, score it on every test split and, optionally, refit the
    support on the OOD-T3 extrapolation trajectories and score again."""
    exp = STLSQExperiment(net, cfg)

    print(f"Fitting STLSQ (threshold={cfg.model.stlsq_threshold})")
    exp.fit_stlsq(data_dict['train_data'], feature_library)
    exp.net.model_phy.print()
    test_results = trainer.test(exp, datamodule=datamodule)

    retrain_test_results = None
    if cfg.training.retrain:
        print("Refitting the STLSQ support by least squares on t3_extra")
        exp.refit_on_support(data_dict['ood_test_data_t3_extra'], feature_library)
        exp.net.model_phy.print()
        retrain_test_results = trainer.test(exp, datamodule=datamodule)

    return exp, retrain_test_results, test_results
