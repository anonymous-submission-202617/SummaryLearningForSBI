"""MMD model for the Gaussian task.

Same architecture as :class:`SMMD_Model`; only the training loss differs.
Uses normalized Gaussian kernels (matching the other benchmark tasks) with
bandwidths ``[10, 5] / n_obs``, where ``n_obs`` is the number of observations
per dataset (see ``sbi_common.losses.mmd_loss``).  At ``n_obs = 50`` the
bandwidths coincide with the historical ``[10, 5] / M`` values (``M = 50``
posterior draws per observation).
"""

from sbi_common.losses import mmd_loss as _mmd_loss

from .smmd import SMMD_Model


def mmd_loss(theta_true, theta_fake, n_obs=50):
    """Multi-scale MMD with normalized kernels and bandwidths ``[10, 5] / n_obs``."""
    return _mmd_loss(theta_true, theta_fake, scale=n_obs, normalize=True)


class MMD_Model(SMMD_Model):
    """Same architecture as SMMD_Model, trained with :func:`mmd_loss`."""

    pass
