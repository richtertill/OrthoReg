"""Container that bundles the symbolic and neural sub-networks.

``HybridExperiment`` (in :mod:`orthoreg.models.experiment`) calls
``self.net.model_phy`` and ``self.net.model_aug`` directly inside its
``derivative_forward`` method and never goes through a ``forward`` of this
class, so the container only registers the two sub-networks as parameters.

The ``method`` and ``options`` arguments are kept on the constructor in case
future code wants to integrate the joint dynamics, but they are unused today.
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
