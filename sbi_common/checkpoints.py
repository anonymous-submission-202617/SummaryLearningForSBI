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


def restore_bayesflow(build_model, observation, parameter_dim, prefix, *, legacy_path):
    """Restore split BayesFlow weights or a checkpoint from an earlier release."""
    summary_path = prefix.with_name(prefix.name + "_summary.weights.h5")
    inference_path = prefix.with_name(prefix.name + "_inference.pt")
    split_exists = summary_path.is_file() or inference_path.is_file()
    if split_exists and not (summary_path.is_file() and inference_path.is_file()):
        raise FileNotFoundError(f"BayesFlow checkpoint requires both {summary_path} and {inference_path}")

    checkpoint = torch.load(inference_path, map_location="cpu", weights_only=True) if split_exists else None
    model = build_model(checkpoint)
    build_bayesflow_for_weights(model, observation, parameter_dim)
    if checkpoint is None:
        if legacy_path.suffix == ".npz":
            with np.load(legacy_path) as archive:
                model.set_weights([archive[f"arr_{i}"] for i in range(len(archive.files))])
        else:
            model.load_weights(legacy_path)
        return model

    standardization = checkpoint["standardization"]
    if standardization.keys() != model.standardize_layers.keys():
        raise ValueError(f"BayesFlow standardization layers do not match {inference_path}")
    model.summary_network.load_weights(summary_path)
    model.inference_network.set_weights([value.numpy() for value in checkpoint["inference_weights"]])
    for name, layer in model.standardize_layers.items():
        layer.set_weights([value.numpy() for value in standardization[name]])
    return model


def save_bayesflow(model, prefix, *, theta_scaling=None):
    """Save the summary and flow separately, including inference preprocessing."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model.summary_network.save_weights(prefix.with_name(prefix.name + "_summary.weights.h5"))
    # Keras-generated torch parameter names change with model construction order.
    checkpoint = {
        "inference_weights": [torch.from_numpy(value) for value in model.inference_network.get_weights()],
        "standardization": {
            name: [torch.from_numpy(value) for value in layer.get_weights()]
            for name, layer in (model.standardize_layers or {}).items()
        },
    }
    if theta_scaling is not None:
        checkpoint["theta_scaling"] = [torch.from_numpy(value) for value in theta_scaling]
    torch.save(checkpoint, prefix.with_name(prefix.name + "_inference.pt"))


def build_sbi_estimator(builder, observation, prior):
    """Infer network shapes; the checkpoint replaces all fitted parameters and buffers."""
    theta = torch.stack((prior.low, prior.high)).detach().cpu()
    x = observation.detach().cpu()
    return builder(theta, torch.cat((x, x + 1), dim=0))
