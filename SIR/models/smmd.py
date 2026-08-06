"""SMMD model for the SIR task."""

import torch
import torch.nn as nn

from sbi_common.losses import sliced_mmd_loss as _sliced_mmd_loss
from sbi_common.nn import (
    RMSNorm,
    Generator,
    SMMD_Model as _SMMD_Model,
    init_normal_std_02 as _init_normal_0_2,
)


class TimeSeriesSummaryNet(nn.Module):
    """1D-CNN summary network for time-series observations (SIR variant)."""

    def __init__(self, input_dim=2, output_dim=10, hidden_dim=64):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.post_conv_norm = RMSNorm(hidden_dim * 2)
        self.apply(_init_normal_0_2)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        features = self.conv_net(x)
        features = features.flatten(1)
        features = self.post_conv_norm(features)
        return self.fc(features)


# Alias for backward compatibility (and usage in DNNABC/SMMD).
SummaryNet = TimeSeriesSummaryNet


class SMMD_Model(_SMMD_Model):
    """SIR SMMD model: CNN summary + 4-layer conditional generator."""

    def __init__(self, summary_dim=10, d=2, d_x=1, n=80):
        super().__init__(
            summary_net=SummaryNet(input_dim=d_x, output_dim=summary_dim),
            generator=Generator(z_dim=d, stats_dim=summary_dim, out_dim=d, hidden_layers=4),
            d=d,
        )


def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_time_steps=80):
    """Sliced MMD with bandwidth ``1 / n_time_steps`` (see ``sbi_common.losses``)."""
    return _sliced_mmd_loss(theta_true, theta_fake, num_slices=num_slices, scale=n_time_steps)

