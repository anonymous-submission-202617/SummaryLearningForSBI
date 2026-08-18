"""MMD model for the Gaussian task.

Same architecture as :class:`SMMD_Model`; only the training loss differs.
Uses normalized Gaussian kernels (matching the other benchmark tasks) with a
mixture of bandwidths scaled by ``n_obs``, the number of observations per
dataset (see ``sbi_common.losses.mmd_loss``).
"""

from sbi_common.losses import mmd_loss as _mmd_loss

from .smmd import SMMD_Model


def mmd_loss(theta_true, theta_fake, n_obs=50):
    """Multi-scale MMD with normalized kernels and a mixture of bandwidths
    scaled by ``n_obs`` (see ``sbi_common.losses.mmd_loss``)."""
    return _mmd_loss(theta_true, theta_fake, scale=n_obs, normalize=True)


class MMD_Model(SMMD_Model):
    """Same architecture as SMMD_Model, trained with :func:`mmd_loss`."""

    pass
