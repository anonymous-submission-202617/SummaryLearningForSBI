"""DNNABC for the Radio (Turin) task.

Trains on log-transformed theta targets and chunks the forward pass during
rejection sampling to bound peak memory.
"""

import os

import torch
import torch.nn as nn

from sbi_common.dnnabc import (
    DNNABC_Model as _DNNABC_Model,
    abc_rejection_sampling as _abc_rejection_sampling,
    train_dnnabc as _train_dnnabc,
)

from .smmd import SummaryNet


class DNNABC_Model(_DNNABC_Model):
    """Regressor mapping observations to log-theta via the Turin summary network."""

    def __init__(self, d=4, d_x=1):
        summary = SummaryNet(input_dim=d_x, output_dim=32)
        head = nn.Linear(32, d)
        super().__init__(summary_net=nn.Sequential(summary, head), d=d)


def _log_theta(theta):
    return torch.log(torch.clamp(theta, min=1e-30))


def train_dnnabc(model, train_loader, epochs, device):
    """Train with AdamW (lr=1e-3), cosine scheduling and log-theta targets."""
    return _train_dnnabc(
        model,
        train_loader,
        epochs,
        device,
        lr=1e-3,
        optimizer="adamw",
        use_cosine_scheduler=True,
        theta_target_transform=_log_theta,
    )


def abc_rejection_sampling(model, x_obs, task, n_samples=1000, n_pool=100000, batch_size=1024, device="cpu"):
    """ABC rejection sampling with chunked forward passes (Radio defaults)."""
    return _abc_rejection_sampling(
        model,
        x_obs,
        task,
        n_samples=n_samples,
        n_pool=n_pool,
        batch_size=batch_size,
        device=device,
        forward_chunk=int(os.environ.get("RADIO_DNNABC_FORWARD_CHUNK", "256")),
    )
