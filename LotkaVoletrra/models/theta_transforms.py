import numpy as np
import torch


def _validate_bounds(lower, upper):
    if lower is None or upper is None:
        raise ValueError("Both lower and upper bounds are required for theta transforms.")


def scale_theta_np(theta, lower, upper):
    _validate_bounds(lower, upper)
    theta = np.asarray(theta, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    span = np.maximum(upper - lower, 1e-6)
    return 2.0 * (theta - lower) / span - 1.0


def unscale_theta_np(theta_scaled, lower, upper, clip=True):
    _validate_bounds(lower, upper)
    theta_scaled = np.asarray(theta_scaled, dtype=np.float32)
    if clip:
        theta_scaled = np.clip(theta_scaled, -1.0, 1.0)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    span = np.maximum(upper - lower, 1e-6)
    return lower + 0.5 * (theta_scaled + 1.0) * span


def  scale_theta_torch(theta, lower, upper):
    _validate_bounds(lower, upper)
    lower = torch.as_tensor(lower, dtype=theta.dtype, device=theta.device)
    upper = torch.as_tensor(upper, dtype=theta.dtype, device=theta.device)
    span = torch.clamp(upper - lower, min=1e-6)
    return 2.0 * (theta - lower) / span - 1.0


def unscale_theta_torch(theta_scaled, lower, upper, clip=True):
    _validate_bounds(lower, upper)
    if clip:
        theta_scaled = torch.clamp(theta_scaled, -1.0, 1.0)
    lower = torch.as_tensor(lower, dtype=theta_scaled.dtype, device=theta_scaled.device)
    upper = torch.as_tensor(upper, dtype=theta_scaled.dtype, device=theta_scaled.device)
    span = torch.clamp(upper - lower, min=1e-6)
    return lower + 0.5 * (theta_scaled + 1.0) * span
