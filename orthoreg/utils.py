"""Tiny utility helpers shared across the package."""

from typing import Any, Optional

from torch.nn import init


def get_wandb() -> Optional[Any]:
    """Return the wandb module if installed, otherwise None."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def init_weights(net, init_type="normal", init_gain=0.01):
    """Initialise the weights of a torch ``nn.Module``.

    Parameters
    ----------
    net : torch.nn.Module
        Module whose ``Conv*`` and ``Linear`` weights should be initialised.
    init_type : str
        One of ``normal``, ``xavier``, ``kaiming``, ``orthogonal``,
        ``default``.
    init_gain : float
        Standard deviation (``normal``) or gain (``xavier`` /
        ``orthogonal``).
    """

    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            "Conv" in classname or "Linear" in classname
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "default":
                pass
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif "BatchNorm" in classname:
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(_init)
