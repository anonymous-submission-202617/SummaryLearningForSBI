"""Utilities for restoring the released model checkpoints."""

from pathlib import Path

import numpy as np
import torch


def released_weights(script_dir, task):
    return Path(script_dir).parent / "WEIGHTS" / f"{task}_weights"


def restore_torch(model, path, device):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def build_bayesflow_for_weights(model, observation, parameter_dim):
    x = np.asarray(observation, dtype=np.float32)[None, ...]
    with torch.no_grad():
        model.summary_network(x)
        model.build({"summary_variables": x.shape, "inference_variables": (1, parameter_dim)})


def build_sbi_estimator(builder, observation, prior):
    """Infer network shapes; the checkpoint replaces all fitted parameters and buffers."""
    theta = torch.stack((prior.low, prior.high)).detach().cpu()
    x = observation.detach().cpu()
    return builder(theta, torch.cat((x, x + 1), dim=0))
