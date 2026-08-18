import argparse
import json
import os
import sys
import time
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

# Make the experiment package and the repository root importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from data_generation import GaussianTask, TRUE_PARAMS, generate_dataset
from models.td_npse import FullSingleObservationDataset, TDNPSE_Model, train_td_npse
from utilities import compute_mmd_metric, compute_swd_metric, run_gaussian_posterior_mcmc, run_folded_nuts_true_posterior


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_select_device(requested: str) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Auto-detected CUDA acceleration. Device: {device} ({torch.cuda.get_device_name(0)})")
            return device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✅ Auto-detected MPS (Apple Silicon) acceleration. Device: {device}")
            return device
        device = torch.device("cpu")
        print(f"ℹ️ Auto-detected CPU fallback. Device: {device}")
        return device

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested)
            idx = device.index if device.index is not None else 0
            print(f"✅ Using requested CUDA device: {device} ({torch.cuda.get_device_name(idx)})")
            return device
        print(f"Warning: requested device '{requested}' but CUDA is not available. Falling back to auto.")
        return auto_select_device("auto")

    if requested == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✅ Using requested MPS device: {device}")
            return device
        print("Warning: requested device 'mps' but MPS is not available. Falling back to auto.")
        return auto_select_device("auto")

    if requested == "cpu":
        device = torch.device("cpu")
        print(f"ℹ️ Using requested CPU device: {device}")
        return device

    print(f"Warning: unsupported device='{requested}'. Expected auto/cpu/mps/cuda[:k]. Falling back to auto.")
    return auto_select_device("auto")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    items = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def parse_optional_int_list(text: str) -> Tuple[int, ...]:
    text = str(text or "").strip()
    if not text:
        return ()
    return tuple(parse_int_list(text))


def invert_stereo_projection(x_obs_3d: np.ndarray) -> np.ndarray:
    x_obs_3d = np.asarray(x_obs_3d, dtype=np.float64)
    if x_obs_3d.ndim != 2 or x_obs_3d.shape[1] != 3:
        raise ValueError(f"Expected x_obs_3d shape (n, 3), got {x_obs_3d.shape}")
    X_comp = x_obs_3d[:, 0]
    Y_comp = x_obs_3d[:, 1]
    Z_comp = x_obs_3d[:, 2]
    denom = 1.0 - Z_comp
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    x_2d = X_comp / denom
    y_2d = Y_comp / denom
    return np.stack([x_2d, y_2d], axis=-1).astype(np.float64)


def plot_loss(loss_history: List[float], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(np.asarray(loss_history, dtype=np.float64), linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("DSM Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_marginals(
    true_samples: np.ndarray,
    approx_samples: np.ndarray,
    theta_true: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    ensure_dir(output_path.parent)
    sns.set_theme(style="whitegrid")
    true_samples = np.asarray(true_samples, dtype=np.float64)
    approx_samples = np.asarray(approx_samples, dtype=np.float64)
    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    d = int(true_samples.shape[1])
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 3.6))
    if d == 1:
        axes = [axes]

    for dim in range(d):
        ax = axes[dim]
        sns.kdeplot(x=true_samples[:, dim], ax=ax, color="green", linewidth=2.4, linestyle="-.", label="True")
        sns.kdeplot(x=approx_samples[:, dim], ax=ax, color="#1f77b4", linewidth=2.2, label="TD-NPSE (GAUSS)")
        ax.axvline(theta_true[dim], color="black", linestyle="--", linewidth=1.4, label="theta_true")
        ax.set_title(rf"$\theta_{{{dim+1}}}$")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_theta34_scatter(true_samples: np.ndarray, approx_samples: np.ndarray, theta_true: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    sns.set_theme(style="whitegrid")
    true_samples = np.asarray(true_samples, dtype=np.float64)
    approx_samples = np.asarray(approx_samples, dtype=np.float64)
    theta_true = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(true_samples[:, 2], true_samples[:, 3], s=10, alpha=0.18, color="green")
    axes[0].scatter([theta_true[2]], [theta_true[3]], color="red", marker="x", s=110, linewidths=2.0)
    axes[0].set_title("True posterior (MCMC)")
    axes[0].set_xlabel(r"$\theta_3$")
    axes[0].set_ylabel(r"$\theta_4$")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(approx_samples[:, 2], approx_samples[:, 3], s=10, alpha=0.18, color="#1f77b4")
    axes[1].scatter([theta_true[2]], [theta_true[3]], color="red", marker="x", s=110, linewidths=2.0)
    axes[1].set_title("TD-NPSE (GAUSS)")
    axes[1].set_xlabel(r"$\theta_3$")
    axes[1].set_ylabel(r"$\theta_4$")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pairplot_single(samples: np.ndarray, output_path: Path, title: str, color: str) -> None:
    ensure_dir(output_path.parent)
    samples = np.asarray(samples, dtype=np.float64)
    df = pd.DataFrame(samples, columns=[f"theta{i+1}" for i in range(samples.shape[1])])
    sns.set_theme(style="white")
    sig = inspect.signature(sns.pairplot)
    kwargs = {
        "diag_kind": "kde",
        "plot_kws": {"s": 8, "alpha": 0.22, "color": color},
        "diag_kws": {"color": color, "linewidth": 2.0},
    }
    if "corner" in sig.parameters:
        kwargs["corner"] = True
    g = sns.pairplot(df, **kwargs)
    g.fig.suptitle(title, y=1.02)
    g.fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def pairplot_overlay(true_samples: np.ndarray, approx_samples: np.ndarray, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    true_samples = np.asarray(true_samples, dtype=np.float64)
    approx_samples = np.asarray(approx_samples, dtype=np.float64)
    df_true = pd.DataFrame(true_samples, columns=[f"theta{i+1}" for i in range(true_samples.shape[1])])
    df_true["source"] = "True"
    df_approx = pd.DataFrame(approx_samples, columns=[f"theta{i+1}" for i in range(approx_samples.shape[1])])
    df_approx["source"] = "TD-NPSE"
    df = pd.concat([df_true, df_approx], axis=0, ignore_index=True)
    sns.set_theme(style="white")
    sig = inspect.signature(sns.pairplot)
    kwargs = {
        "hue": "source",
        "diag_kind": "kde",
        "plot_kws": {"s": 8, "alpha": 0.18},
        "diag_kws": {"linewidth": 2.0},
        "palette": {"True": "green", "TD-NPSE": "#1f77b4"},
    }
    if "corner" in sig.parameters:
        kwargs["corner"] = True
    g = sns.pairplot(df, **kwargs)
    g.fig.suptitle(title, y=1.02)
    g.fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def subsample(samples: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    samples = np.asarray(samples)
    if max_points is None or int(max_points) <= 0:
        return samples
    max_points = int(max_points)
    if samples.shape[0] <= max_points:
        return samples
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(samples.shape[0], size=max_points, replace=False)
    return samples[idx]


def theta34_diagnostics(samples: np.ndarray) -> Dict[str, float]:
    samples = np.asarray(samples, dtype=np.float64)
    theta3 = samples[:, 2]
    theta4 = samples[:, 3]
    quad = {
        "++": int(np.sum((theta3 >= 0.0) & (theta4 >= 0.0))),
        "+-": int(np.sum((theta3 >= 0.0) & (theta4 < 0.0))),
        "-+": int(np.sum((theta3 < 0.0) & (theta4 >= 0.0))),
        "--": int(np.sum((theta3 < 0.0) & (theta4 < 0.0))),
    }
    return {
        "theta3_std": float(np.std(theta3)),
        "theta4_std": float(np.std(theta4)),
        "theta3_abs_mean": float(np.mean(np.abs(theta3))),
        "theta4_abs_mean": float(np.mean(np.abs(theta4))),
        "quadrant_pp": quad["++"],
        "quadrant_pn": quad["+-"],
        "quadrant_np": quad["-+"],
        "quadrant_nn": quad["--"],
    }


def summarize_metrics_by_n(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in raw_df.columns
        if pd.api.types.is_numeric_dtype(raw_df[col]) and col not in {"trial", "metric_seed"}
    ]
    rows = []
    for n_obs, group in raw_df.groupby("n_obs", sort=True):
        row = {"n_obs": int(n_obs), "n_trials": int(group.shape[0])}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_median"] = float(group[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_trial_metrics(
    model: TDNPSE_Model,
    task: GaussianTask,
    theta_true: np.ndarray,
    x_obs_3d: np.ndarray,
    n_samples_posterior: int,
    metric_seed: int,
    n_draws: int,
    n_tune: int,
    chains: int,
    proposal_scale_coeff: float,
    swd_n_projections: int,
    swd_max_samples: int,
    metric_max_samples: int,
    method: str = "folded_nuts",
    target_accept: float = 0.9,
) -> Dict[str, float]:
    t_true0 = time.time()
    true_samples = compute_true_posterior_samples(
        task=task,
        x_obs_3d=x_obs_3d,
        seed=int(metric_seed),
        n_draws=int(n_draws),
        n_tune=int(n_tune),
        chains=int(chains),
        proposal_scale_coeff=float(proposal_scale_coeff),
        method=str(method),
        target_accept=float(target_accept),
    )
    true_time = float(time.time() - t_true0)

    t_app0 = time.time()
    approx_samples = model.sample_posterior(x_obs_3d, n_samples=int(n_samples_posterior))
    approx_time = float(time.time() - t_app0)

    rng = np.random.RandomState(int(metric_seed))
    approx_samples_np = np.asarray(approx_samples, dtype=np.float32)
    true_samples_np = np.asarray(true_samples, dtype=np.float32)
    n_common = min(
        int(metric_max_samples),
        int(swd_max_samples),
        int(approx_samples_np.shape[0]),
        int(true_samples_np.shape[0]),
    )
    if n_common >= 2:
        idx_approx = rng.choice(approx_samples_np.shape[0], size=n_common, replace=False)
        idx_true = rng.choice(true_samples_np.shape[0], size=n_common, replace=False)
        approx_sub = approx_samples_np[idx_approx]
        true_sub = true_samples_np[idx_true]
    else:
        approx_sub = approx_samples_np
        true_sub = true_samples_np

    t_metric0 = time.time()
    mmd_val = float(compute_mmd_metric(approx_sub, true_sub))
    swd_scale = float(np.sqrt(float(task.n)))
    swd_val = float(
        compute_swd_metric(
            approx_sub * swd_scale,
            true_sub * swd_scale,
            n_projections=int(swd_n_projections),
            max_samples=int(swd_max_samples),
            seed=int(metric_seed),
        )
    )
    metric_time = float(time.time() - t_metric0)

    row = {
        "proposal_scale": float(proposal_scale_coeff) / float(np.sqrt(float(task.n))),
        "n_metric_samples": int(n_common),
        "mmd": float(mmd_val),
        "swd": float(swd_val),
        "swd_scale": float(swd_scale),
        "true_posterior_time_sec": float(true_time),
        "td_npse_sampling_time_sec": float(approx_time),
        "metric_time_sec": float(metric_time),
        "eval_time_sec": float(true_time + approx_time + metric_time),
    }
    row.update(theta34_diagnostics(approx_sub))
    return row


def compute_true_posterior_samples(
    task: GaussianTask,
    x_obs_3d: np.ndarray,
    seed: int,
    n_draws: int,
    n_tune: int,
    chains: int,
    proposal_scale_coeff: float,
    method: str = "folded_nuts",
    target_accept: float = 0.9,
    n_per_quad: int = 250,
) -> np.ndarray:
    x_obs_2d = invert_stereo_projection(x_obs_3d)
    if str(method).lower() == "folded_nuts":
        return run_folded_nuts_true_posterior(
            x_obs_2d,
            draws=int(n_draws),
            tune=int(n_tune),
            chains=int(chains),
            target_accept=float(target_accept),
            random_seed=int(seed),
            n_per_quad=int(n_per_quad),
        )
    proposal_scale = float(proposal_scale_coeff) / float(np.sqrt(float(x_obs_2d.shape[0])))
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        samples = run_gaussian_posterior_mcmc(
            x_obs_2d=x_obs_2d,
            task=task,
            n_draws=int(n_draws),
            n_tune_chain=int(n_tune),
            chains=int(chains),
            proposal_scale=float(proposal_scale),
        )
    finally:
        np.random.set_state(state)
    return samples


def build_td_model_kwargs(
    task: GaussianTask,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    td_cfg: Dict,
) -> Dict:
    theta_train = np.asarray(theta_train, dtype=np.float32)
    x_train = np.asarray(x_train, dtype=np.float32)
    theta_mean = theta_train.mean(axis=0).astype(np.float32)
    theta_std = np.clip(theta_train.std(axis=0).astype(np.float32), 1e-6, None)
    x_flat = x_train.reshape(-1, x_train.shape[-1])
    x_mean = x_flat.mean(axis=0).astype(np.float32)
    x_std = np.clip(x_flat.std(axis=0).astype(np.float32), 1e-6, None)

    return dict(
        theta_dim=int(theta_train.shape[1]),
        x_dim=int(x_train.shape[-1]),
        hidden_dim=int(td_cfg["hidden_dim"]),
        num_layers=int(td_cfg["num_layers"]),
        hidden_features=td_cfg.get("hidden_features", None),
        time_frequencies=int(td_cfg["time_frequencies"]),
        num_diffusion_steps=int(td_cfg["num_diffusion_steps"]),
        sample_steps=int(td_cfg["sample_steps"]),
        sampler_type=str(td_cfg["sampler_type"]),
        cov_steps=int(td_cfg["cov_steps"]),
        cov_samples=int(td_cfg["cov_samples"]),
        ddim_eta=float(td_cfg["ddim_eta"]),
        cov_reg=float(td_cfg["cov_reg"]),
        covariance_scale=float(td_cfg["covariance_scale"]),
        cov_mode=str(td_cfg["cov_mode"]),
        clip_denoised=bool(td_cfg["clip_denoised"]),
        prior_lower=task.lower,
        prior_upper=task.upper,
        prior_type=task.prior_type,
        theta_mean=theta_mean,
        theta_std=theta_std,
        x_mean=x_mean,
        x_std=x_std,
        uniform_prior_scale=float(td_cfg["uniform_prior_scale"]),
        tall_score_mode=str(td_cfg["tall_score_mode"]),
        symmetry_sign_flip_dims=td_cfg.get("symmetry_sign_flip_dims", None),
        symmetry_mode_sampling_dims=td_cfg.get("symmetry_mode_sampling_dims", None),
        covariance_symmetry_dims=td_cfg.get("covariance_symmetry_dims", None),
    )


def build_td_model(
    task: GaussianTask,
    theta_train: np.ndarray,
    x_train: np.ndarray,
    td_cfg: Dict,
) -> TDNPSE_Model:
    model_kwargs = build_td_model_kwargs(task=task, theta_train=theta_train, x_train=x_train, td_cfg=td_cfg)
    return TDNPSE_Model(**model_kwargs)


def save_td_checkpoint(
    checkpoint_path: Path,
    model: TDNPSE_Model,
    model_kwargs: Dict,
    td_cfg: Dict,
    train_config: Dict,
    train_time_sec: float,
    epochs_ran: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "td_cfg": td_cfg,
        "train_config": train_config,
        "train_time_sec": float(train_time_sec),
        "epochs_ran": int(epochs_ran),
    }
    torch.save(payload, checkpoint_path)


def load_td_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> Tuple[TDNPSE_Model, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model = TDNPSE_Model(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if map_location is not None:
        model = model.to(map_location)
    return model, checkpoint


def validate_td_checkpoint_reload(checkpoint_path: Path, original_model: TDNPSE_Model) -> Dict[str, float | bool]:
    reloaded_model, checkpoint = load_td_checkpoint(checkpoint_path, map_location="cpu")

    all_match = True
    max_abs_diff = 0.0
    original_state = original_model.state_dict()
    reloaded_state = reloaded_model.state_dict()
    for key in original_state.keys():
        orig = original_state[key].detach().cpu()
        reload = reloaded_state[key].detach().cpu()
        diff = torch.max(torch.abs(orig - reload)).item()
        max_abs_diff = max(max_abs_diff, float(diff))
        if not torch.equal(orig, reload):
            all_match = False

    rng = np.random.RandomState(0)
    x_dim = int(checkpoint["model_kwargs"]["x_dim"])
    theta_dim = int(checkpoint["model_kwargs"]["theta_dim"])
    theta_t = torch.from_numpy(rng.randn(4, theta_dim).astype(np.float32))
    x = torch.from_numpy(rng.randn(4, x_dim).astype(np.float32))
    t = torch.linspace(0.1, 0.9, steps=4, dtype=torch.float32)
    original_model_cpu = TDNPSE_Model(**checkpoint["model_kwargs"])
    original_model_cpu.load_state_dict(original_state)
    original_model_cpu.eval()
    reloaded_model.eval()
    with torch.no_grad():
        y_orig = original_model_cpu(theta_t, x, t)
        y_reload = reloaded_model(theta_t, x, t)
    forward_max_abs_diff = float(torch.max(torch.abs(y_orig - y_reload)).item())

    return {
        "reload_param_match": bool(all_match),
        "reload_param_max_abs_diff": float(max_abs_diff),
        "reload_forward_max_abs_diff": float(forward_max_abs_diff),
        "checkpoint_epochs_ran": int(checkpoint["epochs_ran"]),
        "checkpoint_train_time_sec": float(checkpoint["train_time_sec"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("GAUSSIAN_DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--train-n-obs", type=int, default=1)
    parser.add_argument("--eval-n-obs-list", default="1,15,25,50,100")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", default="cosine")
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping")
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--min-nb-epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--time-frequencies", type=int, default=3)
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=800)
    parser.add_argument("--sampler-type", default="ddim")
    parser.add_argument("--cov-steps", type=int, default=200)
    parser.add_argument("--cov-samples", type=int, default=2000)
    parser.add_argument("--ddim-eta", type=float, default=0.8)
    parser.add_argument("--cov-reg", type=float, default=1e-3)
    parser.add_argument("--covariance-scale", type=float, default=1.0)
    parser.add_argument("--cov-mode", default="GAUSS")
    parser.add_argument("--clip-denoised", action="store_true", default=True)
    parser.add_argument("--no-clip-denoised", action="store_false", dest="clip_denoised")
    parser.add_argument("--uniform-prior-scale", type=float, default=2.0)
    parser.add_argument("--tall-score-mode", default="gauss_clip")
    parser.add_argument("--symmetry-mode-sampling-dims", default="2,3")
    parser.add_argument("--covariance-symmetry-dims", default="")
    parser.add_argument("--n-samples-posterior", type=int, default=1000)
    parser.add_argument("--pairplot-max-points", type=int, default=1500)
    parser.add_argument("--true-method", default="folded_nuts")
    parser.add_argument("--true-draws", type=int, default=250)
    parser.add_argument("--true-tune", type=int, default=200)
    parser.add_argument("--true-chains", type=int, default=4)
    parser.add_argument("--true-target-accept", type=float, default=0.9)
    parser.add_argument("--true-proposal-scale-coeff", type=float, default=0.55)
    parser.add_argument("--metric-num-trials", type=int, default=10)
    parser.add_argument("--metric-max-samples", type=int, default=1000)
    parser.add_argument("--swd-n-projections", type=int, default=1000)
    parser.add_argument("--swd-max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="results/td_npse_singleobs_train_pairplots")
    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_select_device(args.device)

    td_cfg = {
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "time_frequencies": int(args.time_frequencies),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "sample_steps": int(args.sample_steps),
        "sampler_type": str(args.sampler_type),
        "cov_steps": int(args.cov_steps),
        "cov_samples": int(args.cov_samples),
        "ddim_eta": float(args.ddim_eta),
        "cov_reg": float(args.cov_reg),
        "covariance_scale": float(args.covariance_scale),
        "cov_mode": str(args.cov_mode),
        "clip_denoised": bool(args.clip_denoised),
        "uniform_prior_scale": float(args.uniform_prior_scale),
        "tall_score_mode": str(args.tall_score_mode),
        "lr_schedule": str(args.lr_schedule),
        "eta_min": float(args.eta_min),
        "validation_split": float(args.validation_split),
        "early_stopping": bool(args.early_stopping),
        "patience": int(args.patience),
        "min_nb_epochs": int(args.min_nb_epochs),
        "learning_rate": float(args.learning_rate),
        "training_batch_size": int(args.training_batch_size),
        "symmetry_mode_sampling_dims": list(parse_optional_int_list(args.symmetry_mode_sampling_dims)),
        "covariance_symmetry_dims": list(parse_optional_int_list(args.covariance_symmetry_dims)),
    }

    exp_name = (
        f"ntrain{int(args.n_train)}_trainobs{int(args.train_n_obs)}_e{int(args.epochs)}_seed{int(args.seed)}"
    )
    output_dir = (Path(__file__).resolve().parent / args.output_dir / exp_name).resolve()
    plots_dir = output_dir / "plots"
    artifacts_dir = output_dir / "artifacts"
    tables_dir = output_dir / "tables"
    ensure_dir(plots_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(tables_dir)

    with open(artifacts_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "n_train": int(args.n_train),
                "train_n_obs": int(args.train_n_obs),
                "eval_n_obs_list": parse_int_list(args.eval_n_obs_list),
                "seed": int(args.seed),
                "td_cfg": td_cfg,
                "true_posterior": {
                    "draws": int(args.true_draws),
                    "tune": int(args.true_tune),
                    "chains": int(args.true_chains),
                    "proposal_scale_coeff": float(args.true_proposal_scale_coeff),
                },
                "metric_eval": {
                    "num_trials": int(args.metric_num_trials),
                    "metric_max_samples": int(args.metric_max_samples),
                    "swd_n_projections": int(args.swd_n_projections),
                    "swd_max_samples": int(args.swd_max_samples),
                },
                "n_samples_posterior": int(args.n_samples_posterior),
                "device": str(device),
            },
            f,
            indent=2,
        )

    print(f"Output directory: {output_dir}")
    print("=== Step 1: Generate training data (single-observation per theta) ===")
    task_train = GaussianTask(n=int(args.train_n_obs))
    theta_train, x_train = generate_dataset(task_train, n_sims=int(args.n_train), n_obs=int(args.train_n_obs))
    np.save(artifacts_dir / "theta_train.npy", theta_train)
    np.save(artifacts_dir / "x_train.npy", x_train)

    print("=== Step 2: Train TD-NPSE ===")
    model_kwargs = build_td_model_kwargs(task_train, theta_train, x_train, td_cfg)
    model = TDNPSE_Model(**model_kwargs)
    td_dataset = FullSingleObservationDataset(theta_train, x_train)
    td_loader = DataLoader(td_dataset, batch_size=int(args.training_batch_size), shuffle=True)
    t0 = time.time()
    loss_history = train_td_npse(
        model,
        td_loader,
        epochs=int(args.epochs),
        device=device,
        learning_rate=float(args.learning_rate),
        lr_schedule=str(args.lr_schedule),
        eta_min=float(args.eta_min),
        validation_split=float(args.validation_split),
        early_stopping=bool(args.early_stopping),
        patience=int(args.patience),
        min_nb_epochs=int(args.min_nb_epochs),
    )
    train_time = float(time.time() - t0)
    torch.save(model.state_dict(), artifacts_dir / "td_npse_state_dict.pt")
    save_td_checkpoint(
        checkpoint_path=artifacts_dir / "td_npse_checkpoint.pt",
        model=model,
        model_kwargs=model_kwargs,
        td_cfg=td_cfg,
        train_config={
            "n_train": int(args.n_train),
            "train_n_obs": int(args.train_n_obs),
            "seed": int(args.seed),
            "device": str(device),
        },
        train_time_sec=float(train_time),
        epochs_ran=int(len(loss_history)),
    )
    np.save(artifacts_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float64))
    with open(artifacts_dir / "train_time_sec.json", "w") as f:
        json.dump({"train_time_sec": train_time}, f, indent=2)
    reload_check = validate_td_checkpoint_reload(artifacts_dir / "td_npse_checkpoint.pt", model)
    with open(artifacts_dir / "reload_check.json", "w") as f:
        json.dump(reload_check, f, indent=2)
    model, _ = load_td_checkpoint(artifacts_dir / "td_npse_checkpoint.pt", map_location=device)
    plot_loss(loss_history, plots_dir / "loss.png")

    eval_n_list = parse_int_list(args.eval_n_obs_list)
    print("=== Step 3: Use main-experiment TRUE_PARAMS for plotting observations ===")
    theta_true = np.asarray(TRUE_PARAMS, dtype=np.float64).reshape(-1)
    np.save(artifacts_dir / "theta_true.npy", theta_true)
    print(f"theta_true: {theta_true}")

    raw_metric_rows = []
    for n_obs in eval_n_list:
        n_obs = int(n_obs)
        print(f"\n=== Eval: n_obs={n_obs} ===")
        task_eval = GaussianTask(n=n_obs)
        x_obs_3d = np.asarray(task_eval.simulator(theta_true, n_samples=n_obs)[0], dtype=np.float64)
        np.save(artifacts_dir / f"x_obs_n{n_obs}.npy", x_obs_3d)

        t_true0 = time.time()
        true_samples = compute_true_posterior_samples(
            task=task_eval,
            x_obs_3d=x_obs_3d,
            seed=int(args.seed),
            n_draws=int(args.true_draws),
            n_tune=int(args.true_tune),
            chains=int(args.true_chains),
            proposal_scale_coeff=float(args.true_proposal_scale_coeff),
            method=str(args.true_method),
            target_accept=float(args.true_target_accept),
        )
        true_time = float(time.time() - t_true0)
        np.save(artifacts_dir / f"true_posterior_samples_n{n_obs}.npy", true_samples)
        with open(artifacts_dir / f"true_time_sec_n{n_obs}.json", "w") as f:
            json.dump({"true_posterior_time_sec": true_time}, f, indent=2)

        t_app0 = time.time()
        approx_samples = model.sample_posterior(x_obs_3d, n_samples=int(args.n_samples_posterior))
        approx_time = float(time.time() - t_app0)
        np.save(artifacts_dir / f"td_npse_samples_n{n_obs}.npy", approx_samples)
        with open(artifacts_dir / f"td_npse_time_sec_n{n_obs}.json", "w") as f:
            json.dump({"td_npse_sampling_time_sec": approx_time}, f, indent=2)

        true_plot = subsample(true_samples, max_points=int(args.pairplot_max_points), seed=int(args.seed) + 10 * n_obs)
        approx_plot = subsample(
            approx_samples, max_points=int(args.pairplot_max_points), seed=int(args.seed) + 10 * n_obs + 1
        )

        plot_marginals(
            true_samples=true_plot,
            approx_samples=approx_plot,
            theta_true=theta_true,
            output_path=plots_dir / f"marginals_n{n_obs}.png",
            title=f"n_obs={n_obs}",
        )
        plot_theta34_scatter(
            true_samples=true_plot,
            approx_samples=approx_plot,
            theta_true=theta_true,
            output_path=plots_dir / f"theta34_scatter_n{n_obs}.png",
        )
        pairplot_single(true_plot, plots_dir / f"pairplot_true_n{n_obs}.png", f"True posterior (n_obs={n_obs})", "green")
        pairplot_single(
            approx_plot, plots_dir / f"pairplot_td_npse_n{n_obs}.png", f"TD-NPSE (n_obs={n_obs})", "#1f77b4"
        )
        pairplot_overlay(true_plot, approx_plot, plots_dir / f"pairplot_overlay_n{n_obs}.png", f"Overlay (n_obs={n_obs})")

        print(f"=== Metric Eval: n_obs={n_obs}, trials={int(args.metric_num_trials)} ===")
        for trial_idx in range(int(args.metric_num_trials)):
            theta_trial = np.asarray(task_eval.sample_prior(1)[0], dtype=np.float64)
            x_trial = np.asarray(task_eval.simulator(theta_trial, n_samples=n_obs)[0], dtype=np.float64)
            metric_seed = int(args.seed) + int(n_obs) * 100000 + int(trial_idx)
            row = evaluate_trial_metrics(
                model=model,
                task=task_eval,
                theta_true=theta_trial,
                x_obs_3d=x_trial,
                n_samples_posterior=int(args.n_samples_posterior),
                metric_seed=int(metric_seed),
                n_draws=int(args.true_draws),
                n_tune=int(args.true_tune),
                chains=int(args.true_chains),
                proposal_scale_coeff=float(args.true_proposal_scale_coeff),
                method=str(args.true_method),
                target_accept=float(args.true_target_accept),
                swd_n_projections=int(args.swd_n_projections),
                swd_max_samples=int(args.swd_max_samples),
                metric_max_samples=int(args.metric_max_samples),
            )
            row.update(
                {
                    "n_obs": int(n_obs),
                    "trial": int(trial_idx),
                    "theta_true_0": float(theta_trial[0]),
                    "theta_true_1": float(theta_trial[1]),
                    "theta_true_2": float(theta_trial[2]),
                    "theta_true_3": float(theta_trial[3]),
                    "theta_true_4": float(theta_trial[4]),
                    "metric_seed": int(metric_seed),
                    "train_time_sec": float(train_time),
                    "epochs_ran": int(len(loss_history)),
                }
            )
            raw_metric_rows.append(row)
            pd.DataFrame(raw_metric_rows).to_csv(tables_dir / "raw_metrics.csv", index=False)
            print(
                f"n={n_obs} trial={trial_idx + 1}/{int(args.metric_num_trials)} "
                f"MMD={row['mmd']:.6f} SWD={row['swd']:.6f} "
                f"true={row['true_posterior_time_sec']:.2f}s td={row['td_npse_sampling_time_sec']:.2f}s",
                flush=True,
            )

        summary_df = summarize_metrics_by_n(pd.DataFrame(raw_metric_rows))
        summary_df.to_csv(tables_dir / "summary_metrics.csv", index=False)

    raw_df = pd.DataFrame(raw_metric_rows)
    raw_df.to_csv(tables_dir / "raw_metrics.csv", index=False)
    summary_df = summarize_metrics_by_n(raw_df)
    summary_df.to_csv(tables_dir / "summary_metrics.csv", index=False)
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
