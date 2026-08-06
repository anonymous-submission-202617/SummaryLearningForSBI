"""MMD model for the Radio (Turin) task.

Same architecture as :class:`SMMD_Model`; only the training loss differs.
Uses normalized Gaussian kernels scaled by the number of time steps
(see ``sbi_common.losses.mmd_loss``).
"""

from sbi_common.losses import mmd_loss as _mmd_loss

from .smmd import SMMD_Model


def mmd_loss(theta_true, theta_fake, bandwidths=None, n_time_steps=801):
    """Multi-scale MMD with normalized kernels and bandwidths ``[10, 5] / n_time_steps``."""
    return _mmd_loss(
        theta_true,
        theta_fake,
        bandwidths=bandwidths,
        scale=n_time_steps,
        normalize=True,
    )


class MMD_Model(SMMD_Model):
    """Same architecture as SMMD_Model, trained with :func:`mmd_loss`."""

    pass
