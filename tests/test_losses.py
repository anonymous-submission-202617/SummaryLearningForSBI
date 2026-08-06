"""Numerical equivalence tests for the shared MMD family of losses.

Each test compares :mod:`sbi_common.losses` against an independent, explicitly
looped reference implementation of the same estimator to guard against
regressions introduced by the refactoring.
"""

import numpy as np
import math
import torch

from sbi_common.losses import mmd_loss, mixture_sliced_mmd_loss, sliced_mmd_loss


def _set_seed():
    torch.manual_seed(0)
    np.random.seed(0)


def reference_mmd_loss(theta_true, theta_fake, bandwidths, normalize):
    """Looped V-statistic MMD^2 reference (batch, M, d)."""
    batch_size, M, dim = theta_fake.shape
    K = len(bandwidths)
    losses = []
    for b in range(batch_size):
        true = theta_true[b]  # (d,)
        fake = theta_fake[b]  # (M, d)
        tot = 0.0
        for h in bandwidths:
            coef = (2 * np.pi * h) ** (-dim / 2) if normalize else 1.0
            k_gg = 0.0
            for i in range(M):
                for j in range(M):
                    d2 = float(torch.sum((fake[i] - fake[j]) ** 2))
                    k_gg += coef * math.exp(-0.5 * d2 / h)
            k_gg /= M * M
            k_gt = 0.0
            for i in range(M):
                d2 = float(torch.sum((fake[i] - true) ** 2))
                k_gt += coef * math.exp(-0.5 * d2 / h)
            k_gt /= M
            k_tt = coef
            tot += k_gg + k_tt - 2 * k_gt
        # The shared implementation *sums* the multi-scale kernel values
        # (no division by the number of bandwidths).
        losses.append(tot)
    return sum(losses) / batch_size


def test_mmd_loss_matches_reference_unnormalized():
    _set_seed()
    batch, M, d = 3, 7, 4
    theta_true = torch.randn(batch, d)
    theta_fake = torch.randn(batch, M, d)
    scale = 50.0
    got = mmd_loss(theta_true, theta_fake, scale=scale, normalize=False)
    expected = reference_mmd_loss(
        theta_true, theta_fake, [10.0 / scale, 15.0 / scale], normalize=False
    )
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)


def test_mmd_loss_matches_reference_normalized():
    _set_seed()
    batch, M, d = 3, 7, 4
    theta_true = torch.randn(batch, d)
    theta_fake = torch.randn(batch, M, d)
    scale = 151.0
    got = mmd_loss(theta_true, theta_fake, scale=scale, normalize=True)
    expected = reference_mmd_loss(
        theta_true, theta_fake, [10.0 / scale, 15.0 / scale], normalize=True
    )
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)


def test_mmd_loss_default_bandwidths_and_zero_distance():
    _set_seed()
    batch, M, d = 2, 5, 3
    theta_true = torch.randn(batch, d)
    theta_fake = theta_true.unsqueeze(1).expand(batch, M, d).clone()
    # Identical distributions -> MMD^2 == 0 for unnormalized kernels.
    assert torch.allclose(
        mmd_loss(theta_true, theta_fake, scale=50.0, normalize=False),
        torch.tensor(0.0),
        atol=1e-6,
    )


def reference_sliced_mmd_loss(theta_true, theta_fake, unit_vectors, bandwidth):
    """Looped sliced-MMD reference (batch, M, d)."""
    batch_size, M, _ = theta_fake.shape
    losses = []
    for b in range(batch_size):
        proj_t = theta_true[b] @ unit_vectors  # (L,)
        proj_g = theta_fake[b] @ unit_vectors  # (M, L)
        L = unit_vectors.shape[1]
        k_gg = 0.0
        for i in range(M):
            for j in range(M):
                for l in range(L):
                    k_gg += math.exp(-0.5 * (proj_g[i, l] - proj_g[j, l]) ** 2 / bandwidth)
        k_gg /= M * M * L
        k_gt = 0.0
        for i in range(M):
            for l in range(L):
                k_gt += math.exp(-0.5 * (proj_g[i, l] - proj_t[l]) ** 2 / bandwidth)
        k_gt /= M * L
        losses.append(k_gg + 1.0 - 2 * k_gt)
    return sum(losses) / batch_size


def test_sliced_mmd_loss_matches_reference():
    _set_seed()
    batch, M, d, L = 2, 6, 3, 5
    theta_true = torch.randn(batch, d)
    theta_fake = torch.randn(batch, M, d)
    scale = 151.0
    torch.manual_seed(123)
    unit_vectors = torch.randn(d, L)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    torch.manual_seed(123)  # sliced_mmd_loss draws fresh projections internally.
    got = sliced_mmd_loss(theta_true, theta_fake, num_slices=L, scale=scale)
    expected = reference_sliced_mmd_loss(
        theta_true, theta_fake, unit_vectors, 1.0 / scale
    )
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)


def reference_mixture_sliced_mmd_loss(theta_true, theta_fake, bandwidths, unit_vectors):
    batch_size, M, _ = theta_fake.shape
    losses = []
    for b in range(batch_size):
        proj_t = theta_true[b] @ unit_vectors  # (L,)
        proj_g = theta_fake[b] @ unit_vectors  # (M, L)
        L = unit_vectors.shape[1]
        K = len(bandwidths)
        tot = 0.0
        for h in bandwidths:
            coef = 1.0 / float(np.sqrt(2 * np.pi * h))
            k_gg = 0.0
            for i in range(M):
                for j in range(M):
                    for l in range(L):
                        k_gg += coef * math.exp(-0.5 * (proj_g[i, l] - proj_g[j, l]) ** 2 / h)
            k_gg /= M * M * L
            k_gt = 0.0
            for i in range(M):
                for l in range(L):
                    k_gt += coef * math.exp(-0.5 * (proj_g[i, l] - proj_t[l]) ** 2 / h)
            k_gt /= M * L
            tot += k_gg + coef - 2 * k_gt
        losses.append(tot / K)
    return sum(losses) / batch_size


def test_mixture_sliced_mmd_loss_matches_reference():
    _set_seed()
    batch, M, d, L = 2, 6, 3, 5
    theta_true = torch.randn(batch, d)
    theta_fake = torch.randn(batch, M, d)
    scale = 151.0
    bandwidths = [20.0 / scale, 20.0 / scale]
    torch.manual_seed(123)
    unit_vectors = torch.randn(d, L)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    torch.manual_seed(123)  # mixture_sliced_mmd_loss draws fresh projections internally.
    got = mixture_sliced_mmd_loss(
        theta_true, theta_fake, bandwidths=bandwidths, num_slices=L, scale=scale
    )
    expected = reference_mixture_sliced_mmd_loss(theta_true, theta_fake, bandwidths, unit_vectors)
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)
