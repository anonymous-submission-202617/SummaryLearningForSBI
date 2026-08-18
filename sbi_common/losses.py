"""Distance-based losses shared by the SMMD/MMD family of models.

Two MMD variants are implemented:

* :func:`mmd_loss` — multi-scale MMD with Gaussian kernels on raw theta.
* :func:`sliced_mmd_loss` — sliced MMD on random one-dimensional projections.
* :func:`mixture_sliced_mmd_loss` — sliced MMD with a mixture of bandwidths.

The bandwidth scale is the number of observations in a single observed sample
(``n`` for the Gaussian task, the time-series length ``T`` for the
Lotka-Volterra / SIR / Radio tasks).  Following the Bernstein-von Mises
posterior contraction rate (posterior std ~ ``1 / sqrt(scale)``), the default
bandwidths are ``h = k^2 / scale`` (kernel width ``sqrt(h) ~ k / sqrt(scale)``),
so the kernel keeps resolving posterior differences as the sample size grows.
The per-experiment ``models/mmd.py`` / ``models/smmd.py`` wrappers bind the
scale argument so that numerical behaviour is unchanged.
"""

import numpy as np
import torch


def mmd_loss(theta_true, theta_fake, *, bandwidths=None, scale=50, normalize=False):
    """Multi-scale MMD (V-statistic) with Gaussian kernels.

    Args:
        theta_true: (batch, d) observed parameters.
        theta_fake: (batch, M, d) simulated parameters.
        bandwidths: Optional (K,) tensor/list of bandwidths (sigma^2).
            Defaults to a mixture of two Gaussian bandwidths proportional to
            ``1 / scale`` (see the default value in the function body).
        scale: Number of observations per observed sample.  The default
            bandwidths follow the ``k^2 / scale`` family, i.e. kernel widths
            ``sqrt(h) ~ k / sqrt(scale)``, so the kernel keeps resolving
            posterior differences as the sample size grows.
        normalize: If True, weight each kernel by the Gaussian normalization
            coefficient ``(2 * pi * bandwidth)^(-d/2)``.

    Returns:
        Scalar MMD^2 estimate averaged over the batch.
    """
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device

    if bandwidths is None:
        bandwidths = torch.tensor([10.0 / scale, 15.0 / scale], device=device)
    elif not isinstance(bandwidths, torch.Tensor):
        bandwidths = torch.tensor(bandwidths, device=device, dtype=torch.float32)
    else:
        bandwidths = bandwidths.to(device)

    # Reshape for broadcasting: (K, 1, 1, 1).
    bandwidths = bandwidths.view(-1, 1, 1, 1)

    if normalize:
        # bandwidths is sigma^2; coefficient = (2 * pi * sigma^2)^(-d/2).
        coefs = (2 * np.pi * bandwidths) ** (-dim / 2)
    else:
        coefs = None

    # Pairwise squared distances.
    theta_true_exp = theta_true.unsqueeze(1)  # (batch, 1, d)
    dist_sq_GG = torch.cdist(theta_fake, theta_fake, p=2) ** 2  # (batch, M, M)
    dist_sq_GT = torch.cdist(theta_fake, theta_true_exp, p=2) ** 2  # (batch, M, 1)

    # Gaussian kernels (summed over bandwidths without renormalization).
    K_GG = torch.exp(-0.5 * dist_sq_GG.unsqueeze(0) / bandwidths)  # (K, batch, M, M)
    K_GT = torch.exp(-0.5 * dist_sq_GT.unsqueeze(0) / bandwidths)  # (K, batch, M, 1)
    if normalize:
        K_GG = coefs * K_GG
        K_TT = coefs.expand(-1, batch_size, 1, 1)
        K_GT = coefs * K_GT
    else:
        K_TT = torch.ones((bandwidths.shape[0], batch_size, 1, 1), device=device)

    total_K_GG = torch.sum(K_GG, dim=0)  # (batch, M, M)
    total_K_TT = torch.sum(K_TT, dim=0)  # (batch, 1, 1)
    total_K_GT = torch.sum(K_GT, dim=0)  # (batch, M, 1)

    # V-statistic (biased estimator), no M/(M-1) correction.
    mean_K_GG = torch.mean(total_K_GG, dim=(1, 2))
    mean_K_GT = torch.mean(total_K_GT, dim=(1, 2))
    mean_K_TT = torch.mean(total_K_TT, dim=(1, 2))

    return torch.mean(mean_K_GG + mean_K_TT - 2 * mean_K_GT)


def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, scale=50):
    """Sliced MMD with Gaussian kernels on random unit projections.

    Args:
        theta_true: (batch, d) observed parameters.
        theta_fake: (batch, M, d) simulated parameters.
        num_slices: Number of random one-dimensional projections.
        scale: Number of observations per observed sample.  Sets
            ``bandwidth = 1 / scale``, i.e. kernel width ``1 / sqrt(scale)``
            (the ``k = 1`` member of the ``k^2 / scale`` family).

    Returns:
        Scalar loss averaged over the batch.
    """
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device

    # 1. Random projections: (dim, num_slices), normalized columns.
    unit_vectors = torch.randn(dim, num_slices, device=device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)

    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)  # (batch, 1, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)  # (batch, M, L)

    # 2. MMD on the projections with a single Gaussian kernel.
    bandwidth = 1.0 / (1.0 * scale)

    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)  # (batch, M, M, L)
    K_GG = torch.exp(-0.5 * diff_GG.pow(2) / bandwidth)
    loss_GG = torch.mean(K_GG, dim=(1, 2, 3))

    # T vs T: distance 0 -> kernel value 1.
    loss_TT = torch.tensor(1.0, device=device)

    diff_GT = proj_G - proj_T  # (batch, M, L)
    K_GT = torch.exp(-0.5 * diff_GT.pow(2) / bandwidth)
    loss_GT = torch.mean(K_GT, dim=(1, 2))

    loss = loss_GG + loss_TT - 2 * loss_GT
    return torch.mean(loss)


def mixture_sliced_mmd_loss(theta_true, theta_fake, bandwidths=None, num_slices=20, scale=151):
    """Sliced MMD with a mixture of Gaussian kernels (multiple bandwidths).

    Args:
        theta_true: (batch, d) ground-truth parameters.
        theta_fake: (batch, M, d) simulated parameters.
        bandwidths: Optional (K,) tensor/list of bandwidths (sigma^2).
            Defaults to ``[20 / scale, 20 / scale]``.
        num_slices: Number of random projections.
        scale: Number of observations per observed sample, used for the
            default bandwidths.

    Returns:
        Scalar loss averaged over the batch.
    """
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device

    if bandwidths is None:
        bandwidths = [20.0 / scale, 20.0 / scale]
    if not isinstance(bandwidths, torch.Tensor):
        bandwidths = torch.tensor(bandwidths, device=device, dtype=torch.float32)
    else:
        bandwidths = bandwidths.to(device)

    num_kernels = bandwidths.shape[0]
    # coefficient = 1 / sqrt(2 * pi * sigma^2) for each kernel.
    coefs = 1.0 / torch.sqrt(2 * np.pi * bandwidths)  # (K,)

    unit_vectors = torch.randn(dim, num_slices, device=device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)

    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)  # (batch, 1, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)  # (batch, M, L)

    # G vs G.
    diff_sq_GG = (proj_G.unsqueeze(2) - proj_G.unsqueeze(1)).pow(2)  # (batch, M, M, L)
    d_GG = diff_sq_GG.unsqueeze(-1)  # (batch, M, M, L, 1)
    b_view = bandwidths.view(1, 1, 1, 1, num_kernels)
    c_view = coefs.view(1, 1, 1, 1, num_kernels)
    K_GG_all = c_view * torch.exp(-0.5 * d_GG / b_view)
    loss_GG = torch.mean(K_GG_all, dim=(1, 2, 3, 4))  # (batch,)

    # T vs T: distance 0 -> kernel value 1 * coef.
    loss_TT = torch.mean(coefs)

    # G vs T.
    diff_sq_GT = (proj_G - proj_T).pow(2)  # (batch, M, L)
    d_GT = diff_sq_GT.unsqueeze(-1)  # (batch, M, L, 1)
    b_view_GT = bandwidths.view(1, 1, 1, num_kernels)
    c_view_GT = coefs.view(1, 1, 1, num_kernels)
    K_GT_all = c_view_GT * torch.exp(-0.5 * d_GT / b_view_GT)
    loss_GT = torch.mean(K_GT_all, dim=(1, 2, 3))  # (batch,)

    loss = loss_GG + loss_TT - 2 * loss_GT
    return torch.mean(loss)
