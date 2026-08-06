"""SMMD model for the Gaussian task (deep-set summary network)."""

import torch
import torch.nn as nn

from sbi_common.losses import sliced_mmd_loss as _sliced_mmd_loss
from sbi_common.nn import (
    RMSNorm,
    Generator,
    SMMD_Model as _SMMD_Model,
    init_normal_std_02 as _init_normal_0_2,
)


class InvariantModule(nn.Module):
    """Permutation-invariant block: dense layers, mean pooling, dense layers."""

    def __init__(self, settings):
        super().__init__()

        # S1: dense layers before pooling.
        s1_layers = []
        in_dim = settings["input_dim"]
        for _ in range(settings["num_dense_s1"]):
            s1_layers.append(nn.Linear(in_dim, settings["dense_s1_args"]["units"]))
            s1_layers.append(nn.ReLU())
            in_dim = settings["dense_s1_args"]["units"]
        self.s1 = nn.Sequential(*s1_layers)
        self.s1_out_dim = in_dim

        # S2: dense layers after pooling.
        s2_layers = []
        in_dim = self.s1_out_dim
        for _ in range(settings["num_dense_s2"]):
            s2_layers.append(nn.Linear(in_dim, settings["dense_s2_args"]["units"]))
            s2_layers.append(nn.ReLU())
            in_dim = settings["dense_s2_args"]["units"]
        self.s2 = nn.Sequential(*s2_layers)
        self.output_dim = in_dim

    def forward(self, x):
        # x: (batch, n_points, input_dim)
        x_s1 = self.s1(x)  # (batch, n_points, s1_out_dim)
        x_reduced = torch.mean(x_s1, dim=1)  # (batch, s1_out_dim)
        return self.s2(x_reduced)  # (batch, s2_out_dim)


class EquivariantModule(nn.Module):
    """Equivariant block: invariant features concatenated back to each point."""

    def __init__(self, settings):
        super().__init__()
        self.invariant_module = InvariantModule(settings)

        # S3: dense layers combining original x and invariant features.
        s3_layers = []
        in_dim = settings["input_dim"] + self.invariant_module.output_dim
        for _ in range(settings["num_dense_s3"]):
            s3_layers.append(nn.Linear(in_dim, settings["dense_s3_args"]["units"]))
            s3_layers.append(nn.ReLU())
            in_dim = settings["dense_s3_args"]["units"]
        self.s3 = nn.Sequential(*s3_layers)
        self.output_dim = in_dim

    def forward(self, x):
        # x: (batch, n_points, input_dim)
        batch_size, n_points, _ = x.shape

        out_inv = self.invariant_module(x)  # (batch, inv_out_dim)
        out_inv_rep = out_inv.unsqueeze(1).expand(-1, n_points, -1)
        out_c = torch.cat([x, out_inv_rep], dim=-1)
        return self.s3(out_c)  # (batch, n_points, s3_out_dim)


class SummaryNet(nn.Module):
    """Deep-set summary network: two equivariant blocks, then an invariant head."""

    def __init__(self, input_dim=3, output_dim=10):
        super().__init__()

        settings = dict(
            num_dense_s1=2,
            num_dense_s2=2,
            num_dense_s3=2,
            dense_s1_args={"units": 64},
            dense_s2_args={"units": 64},
            dense_s3_args={"units": 64},
            input_dim=input_dim,
        )

        # Layer 1: equivariant.
        self.equiv1 = EquivariantModule(settings)

        # Layer 2: equivariant (updated input dim).
        settings_l2 = settings.copy()
        settings_l2["input_dim"] = self.equiv1.output_dim
        self.equiv2 = EquivariantModule(settings_l2)

        # Layer 3: invariant.
        settings_l3 = settings.copy()
        settings_l3["input_dim"] = self.equiv2.output_dim
        self.inv = InvariantModule(settings_l3)

        self.post_pool_norm = RMSNorm(self.inv.output_dim)
        self.out_layer = nn.Linear(self.inv.output_dim, output_dim)
        self.apply(_init_normal_0_2)

    def forward(self, x):
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        x = self.post_pool_norm(x)
        return self.out_layer(x)


class SMMD_Model(_SMMD_Model):
    """Gaussian SMMD model: deep-set summary + 3-layer conditional generator."""

    def __init__(self, summary_dim=10, d=5, d_x=3, n=50):
        super().__init__(
            summary_net=SummaryNet(input_dim=d_x, output_dim=summary_dim),
            generator=Generator(z_dim=d, stats_dim=summary_dim, out_dim=d, hidden_layers=3),
            d=d,
        )


def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_obs=50):
    """Sliced MMD with bandwidth ``1 / n_obs`` (see ``sbi_common.losses``)."""
    return _sliced_mmd_loss(theta_true, theta_fake, num_slices=num_slices, scale=n_obs)
