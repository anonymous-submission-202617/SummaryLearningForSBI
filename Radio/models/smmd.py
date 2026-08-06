"""SMMD model for the Radio (Turin) task.

The generator outputs are standardized with per-parameter mean/std; posterior
samples are mapped back to the original scale when a standardizer is set.
"""

import torch
import torch.nn as nn

from sbi_common.losses import sliced_mmd_loss as _sliced_mmd_loss
from sbi_common.nn import (
    RMSNorm,
    Generator,
    SMMD_Model as _SMMD_Model,
    init_normal_std_02 as _init_normal_0_2,
)


class TurinSummary(nn.Module):
    """Convolutional summary network for Turin impulse-response observations."""

    def __init__(
        self,
        input_dim=1,
        output_dim=10,
        hidden_dim=32,
        input_size=None,
    ):
        super().__init__()
        in_ch = int(input_size) if input_size is not None else int(input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 8, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(64, 8, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(8, output_dim)
        self.out_norm = RMSNorm(output_dim)
        self.apply(_init_normal_0_2)

    def forward(self, x):
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim == 2:
            x = x[:, None, :]
        b, n, ns = x.shape
        x = x.reshape(b * n, 1, ns)
        h = self.conv(x).squeeze(-1)
        h = h.reshape(b, n, -1).mean(dim=1)
        y = self.fc(h)
        y = self.out_norm(y)
        return torch.tanh(y)


# Alias for backward compatibility (and usage in DNNABC/SMMD).
SummaryNet = TurinSummary


class SMMD_Model(_SMMD_Model):
    """Radio SMMD model with an optional theta standardizer for posterior samples."""

    def __init__(self, summary_dim=10, d=4, d_x=1, n=801):
        super().__init__(
            summary_net=SummaryNet(input_dim=d_x, output_dim=summary_dim),
            generator=Generator(z_dim=d, stats_dim=summary_dim, out_dim=d, hidden_layers=4),
            d=d,
        )
        self.register_buffer("theta_mean_buf", torch.empty(0))
        self.register_buffer("theta_std_buf", torch.empty(0))

    def set_theta_standardizer(self, theta_mean, theta_std) -> None:
        tm = torch.as_tensor(theta_mean, dtype=torch.float32)
        ts = torch.as_tensor(theta_std, dtype=torch.float32)
        if tm.ndim != 1 or ts.ndim != 1 or tm.shape[0] != int(self.d) or ts.shape[0] != int(self.d):
            raise ValueError(f"Invalid theta_mean/theta_std shapes: mean {tm.shape}, std {ts.shape}, d={self.d}")
        self.theta_mean_buf = tm
        self.theta_std_buf = ts

    def _unstandardize(self, samples):
        if self.theta_mean_buf.numel() == int(self.d) and self.theta_std_buf.numel() == int(self.d):
            return samples * self.theta_std_buf + self.theta_mean_buf
        return samples

    def sample_posterior(self, x_obs, n_samples):
        return super().sample_posterior(x_obs, n_samples, transform=self._unstandardize)


def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_time_steps=801):
    """Sliced MMD with bandwidth ``1 / n_time_steps`` (see ``sbi_common.losses``)."""
    return _sliced_mmd_loss(theta_true, theta_fake, num_slices=num_slices, scale=n_time_steps)
