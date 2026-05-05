"""Container that bundles the symbolic and neural sub-networks.

Why this file is small: in the OLD ``git/hybrid/hybrid/forecasters.py``
``Forecaster.forward`` integrated the combined dynamics with
``torchdiffeq.odeint`` and dispatched to a ``DerivativeEstimator`` that
called both ``model_phy`` and ``model_aug``. None of that was reachable
from the shipping training pipeline: ``HybridExperiment`` (in
:mod:`orthoreg.models.exp`) accesses ``self.net.model_phy`` and
``self.net.model_aug`` directly inside its ``derivative_forward`` method,
never going through ``Forecaster.forward``. Verified by ``grep -rn
'forecaster\\.forward\\|net\\.forward'`` over the OLD
``hybrid/{exp,train,train_baseline,setup,datamodule}.py`` -- the only
producer of those attributes is ``setup_model``; nothing else calls
``forward``.

The ``method`` and ``options`` arguments are kept on the constructor for
parity with the OLD signature in case future code wants to integrate the
joint dynamics, but they are unused today.
"""

import torch.nn as nn


class Forecaster(nn.Module):
    """Holds ``model_phy`` (symbolic) and ``model_aug`` (neural) together.

    The training loop calls ``self.net.model_phy(transformed_y, ...)`` and
    ``self.net.model_aug(transformed_y)`` directly; this class is purely a
    parameter-registration container.
    """

    def __init__(self, model_phy, model_aug, hybrid_setting,
                 method="rk4", options=None):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.hybrid_setting = hybrid_setting
        self.method = method
        self.options = options or {"step_size": 0.01, "max_num_steps": 1000}
