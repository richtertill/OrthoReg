"""Dataset registry for OrthoReg.

Mapping between the OLD ``git/hybrid/hybrid/datasets/__init__.py`` and
this file (kept intentionally minimal):

    OLD key                      NEW key       Class                          Notes
    ---------------------------  ------------  -----------------------------  ------------------------------
    theoretical_pendulum         pendulum      PendulumDataset                renamed (paper's Pendulum, cos/exp/tanh missing)
    lv                           lv            LotkaVolterraDataset           now produces the same 8-tuple split as pendulum/duffing
    sir                          sir           SIREpidemicDataset             same split contract as lv
    duffing                      duffing       DuffingOscillatorDataset       Goring et al. (2024) parameters
    pendulum (DampedPendulum)    --            --                             dropped (paper uses TheoreticalPendulumDataset)
    tweak_add_pendulum           --            --                             dropped (out-of-paper-scope dataset variant)
    tweak_mul_pendulum           --            --                             dropped (out-of-paper-scope dataset variant)
    tweak_add_lv                 --            --                             dropped (out-of-paper-scope)
    tweak_mul_lv                 --            --                             dropped (out-of-paper-scope)
    complex_orthogonal_lv        --            --                             dropped (out-of-paper-scope)
    tweak_add_sir                --            --                             dropped (out-of-paper-scope)
    tweak_mul_sir                --            --                             dropped (out-of-paper-scope)
    tweaked_sir                  --            --                             dropped (out-of-paper-scope)
    underspecified_duffing       --            --                             dropped (not used in the released table)

Each :func:`init_dataloaders` call returns the six-way split used in the
paper (train / id-test / ood-t2 / ood-t3, each with an extrapolation pair)
plus the ground-truth coefficient dictionary needed by the symbolic-
recovery metrics. The contract is identical to the OLD
``param_theoretical_pendulum`` / ``param_duffing`` helpers; the OLD
``param_lv`` / ``param_sir`` returned a 4-tuple, which the new
implementations standardise to the 8-tuple form so all four systems flow
through the same datamodule.
"""

from typing import Optional

import numpy as np

from orthoreg.data.datasets.duffing import DuffingOscillatorDataset
from orthoreg.data.datasets.lv import LotkaVolterraDataset
from orthoreg.data.datasets.pend import PendulumDataset
from orthoreg.data.datasets.sir import SIREpidemicDataset
from orthoreg.paths import DATA_DIR


_DATASET_REGISTRY = {
    "pendulum": PendulumDataset,
    "lv": LotkaVolterraDataset,
    "sir": SIREpidemicDataset,
    "duffing": DuffingOscillatorDataset,
}


def _build_split(dataset_cls, n_samples, times, granularity, sampling_scheme,
                 feature_library, extrapolation_ratio=0.3, cfg=None):
    data_kfold = dataset_cls.get_standard_dataset(
        root=DATA_DIR,
        n_samples=n_samples,
        times=times,
        granularity=granularity,
        sampling_scheme=sampling_scheme,
        extrapolation_ratio=extrapolation_ratio,
        cfg=cfg,
    )
    ((train_data, train_data_extra),
     (id_test_data, id_test_data_extra),
     (ood_test_data_t2, ood_test_data_t2_extra),
     (ood_test_data_t3, ood_test_data_t3_extra)) = data_kfold[0]

    nT = granularity * times
    t = np.linspace(0, int(times), int(nT))
    dataset_instance = dataset_cls(n_samples=n_samples, t=t)
    W_true = dataset_cls.compute_true_W(
        dataset_instance,
        feature_library=feature_library,
        train_data=train_data,
    )
    return (
        (train_data, train_data_extra),
        (id_test_data, id_test_data_extra),
        (ood_test_data_t2, ood_test_data_t2_extra),
        (ood_test_data_t3, ood_test_data_t3_extra),
        W_true,
    )


def init_dataloaders(dataset, feature_library, n_samples, times, granularity,
                     sampling_scheme, buffer_filepath: Optional[str] = None,
                     cfg=None):
    """Build the standard six-way split for ``dataset``.

    ``dataset`` is one of ``pendulum``, ``lv``, ``sir``, ``duffing``.
    """
    try:
        dataset_cls = _DATASET_REGISTRY[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Expected one of {sorted(_DATASET_REGISTRY)}."
        ) from exc
    return _build_split(
        dataset_cls,
        n_samples=n_samples,
        times=times,
        granularity=granularity,
        sampling_scheme=sampling_scheme,
        feature_library=feature_library,
        cfg=cfg,
    )
