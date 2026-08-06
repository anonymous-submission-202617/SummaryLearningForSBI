"""SMMD model for the Lotka-Volterra task.

The generator maps from a bounded (scaled) theta representation; posterior
samples are unscaled back to the original parameter space.
"""

import torch
import torch.nn as nn

from sbi_common.losses import (
    mixture_sliced_mmd_loss as _mixture_sliced_mmd_loss,
    sliced_mmd_loss as _sliced_mmd_loss,
)
from sbi_common.nn import (
    RMSNorm,
    Generator,
    SMMD_Model as _SMMD_Model,
    init_normal_std_02 as _init_normal_0_2,
)

from .theta_transforms import scale_theta_torch, unscale_theta_torch

class TimeSeriesSummaryNet(nn.Module):
    """1D-CNN summary network for time-series observations."""

    def __init__(self, input_dim=2, output_dim=10, hidden_dim=64):
        super().__init__()

        self.conv_net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=10, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=10, padding=2),
            nn.ReLU(),
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.post_conv_norm = RMSNorm(hidden_dim * 2)
        self.apply(_init_normal_0_2)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        features = self.conv_net(x)  # (batch, hidden*2, 1)
        features = features.flatten(1)  # (batch, hidden*2)
        features = self.post_conv_norm(features)
        return self.fc(features)


# Alias for backward compatibility (and usage in DNNABC/SMMD).
SummaryNet = TimeSeriesSummaryNet

class SMMD_Model(_SMMD_Model):
    """Lotka-Volterra SMMD model with bounded (scaled) theta representation."""

    def __init__(self, summary_dim=10, d=4, d_x=2, n=151, theta_lower=None, theta_upper=None):
        super().__init__(
            summary_net=SummaryNet(input_dim=d_x, output_dim=summary_dim),
            generator=Generator(
                z_dim=d,
                stats_dim=summary_dim,
                out_dim=d,
                hidden_layers=4,
                output_activation=nn.Tanh(),
            ),
            d=d,
        )
        if theta_lower is not None and theta_upper is not None:
            theta_lower = torch.as_tensor(theta_lower, dtype=torch.float32).reshape(1, d)
            theta_upper = torch.as_tensor(theta_upper, dtype=torch.float32).reshape(1, d)
        else:
            theta_lower = None
            theta_upper = None
        self.register_buffer("theta_lower", theta_lower)
        self.register_buffer("theta_upper", theta_upper)

    def scale_theta(self, theta):
        if self.theta_lower is None or self.theta_upper is None:
            return theta
        return scale_theta_torch(theta, self.theta_lower, self.theta_upper)

    def unscale_theta(self, theta_scaled):
        if self.theta_lower is None or self.theta_upper is None:
            return theta_scaled
        return unscale_theta_torch(theta_scaled, self.theta_lower, self.theta_upper)

    def sample_posterior(self, x_obs, n_samples):
        return super().sample_posterior(x_obs, n_samples, transform=self.unscale_theta)


def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_time_steps=151):
    """Sliced MMD with bandwidth ``1 / n_time_steps`` (see ``sbi_common.losses``)."""
    return _sliced_mmd_loss(theta_true, theta_fake, num_slices=num_slices, scale=n_time_steps)

