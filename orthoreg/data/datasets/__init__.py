"""Dataset registry for OrthoReg.

The five paper systems, keyed by ``dataset.dataset_name``:

    key              class                         out-of-library terms (K)
    ---------------  ----------------------------  ------------------------------------
    pendulum         PendulumDataset               cos(3 x0), exp(-x0^2), tanh(x1)
    duffing          DuffingOscillatorDataset      c * x0^3
    lotka_volterra   LotkaVolterraDataset          one modulated coupling per equation
    sir              SIREpidemicDataset            one memory-modulated term per equation
    helmholtz        HelmholtzOscillatorDataset    A * cos(Omega * x0)

Each :func:`init_dataloaders` call returns the six-way split used in the
paper (train / id-test / ood-t2 / ood-t3, each with an extrapolation pair)
plus the ground-truth coefficient dictionary of the training split, which
the symbolic-recovery metrics compare against.
"""

from typing import Optional

from orthoreg.data.datasets.duffing import DuffingOscillatorDataset
from orthoreg.data.datasets.helmholtz import HelmholtzOscillatorDataset
from orthoreg.data.datasets.lotka_volterra import LotkaVolterraDataset
from orthoreg.data.datasets.pendulum import PendulumDataset
from orthoreg.data.datasets.sir import SIREpidemicDataset
from orthoreg.paths import DATA_DIR


_DATASET_REGISTRY = {
    "pendulum": PendulumDataset,
    "duffing": DuffingOscillatorDataset,
    "lotka_volterra": LotkaVolterraDataset,
    "sir": SIREpidemicDataset,
    "helmholtz": HelmholtzOscillatorDataset,
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

    W_true = train_data.compute_true_W(
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

    ``dataset`` is one of ``pendulum``, ``duffing``, ``lotka_volterra``,
    ``sir``, ``helmholtz``.
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
