"""Shared neural-network components for the SMMD/MMD family of models.

These modules are used by every benchmark experiment.  Per-experiment
``models/smmd.py`` modules provide the task-specific summary networks and
subclass :class:`SMMD_Model` with the experiment's defaults.
"""

import torch
import torch.nn as nn


def init_normal_std_02(module):
    """Initialize ``nn.Linear``/``nn.Conv1d`` weights with N(0, 0.2), zero biases."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.normal_(module.weight, mean=0.0, std=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# Backward-compatible alias used by some experiment modules.
_init_normal_0_2 = init_normal_std_02


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean centering)."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class Generator(nn.Module):
    """Conditional generator: maps (latent ``z``, summary statistics) to theta.

    The architecture is a stack of ``hidden_layers`` MLP blocks with RMSNorm
    and ReLU, followed by a linear output layer and an optional output
    activation (e.g. ``nn.Tanh`` when theta is trained on a bounded scale).
    """

    def __init__(
        self,
        z_dim=5,
        stats_dim=10,
        out_dim=5,
        hidden_dim=64,
        hidden_layers=3,
        output_activation=None,
    ):
        super().__init__()
        in_dim = z_dim + stats_dim
        layers = [
            nn.Linear(in_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(max(hidden_layers - 1, 0)):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    RMSNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)
        self.apply(init_normal_std_02)

    def forward(self, z, stats):
        # z: (batch, M, z_dim), stats: (batch, stats_dim)
        stats_exp = stats.unsqueeze(1).expand(-1, z.size(1), -1)
        gen_input = torch.cat([z, stats_exp], dim=-1)
        return self.net(gen_input)  # (batch, M, out_dim)


class SMMD_Model(nn.Module):
    """SMMD model: summary network ``T`` plus conditional generator ``G``.

    Subclasses supply the task-specific summary network and generator
    configuration; ``sample_posterior`` optionally applies a ``transform``
    (e.g. unscaling from a bounded or standardized representation).
    """

    def __init__(self, summary_net, generator, d):
        super().__init__()
        self.T = summary_net
        self.G = generator
        self.d = d

    def forward(self, x_obs, z):
        stats = self.T(x_obs)
        return self.G(z, stats)

    def compute_stats(self, x):
        return self.T(x)

    def sample_posterior(self, x_obs, n_samples, transform=None):
        # x_obs: (1, n, d_x) tensor, or numpy array (n, d_x) / (1, n, d_x)
        if not isinstance(x_obs, torch.Tensor):
            device = next(self.parameters()).device
            x_obs = torch.from_numpy(x_obs).float().to(device)
            if x_obs.ndim == 2:
                x_obs = x_obs.unsqueeze(0)

        device = x_obs.device
        with torch.no_grad():
            stats = self.T(x_obs)  # (1, p)
            Z = torch.randn(1, n_samples, self.d, device=device)
            samples = self.G(Z, stats).squeeze(0)  # (n_samples, d)
        if transform is not None:
            samples = transform(samples)
        return samples.cpu().numpy()
