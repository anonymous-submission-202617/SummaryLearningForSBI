"""
Expected coverage / calibration experiment for the SIR benchmark.

This script trains amortized models once, then evaluates calibration by sampling
independent pairs (theta_i, x_i) from the joint distribution p(theta, x):

    theta_i ~ p(theta)
    x_i ~ p(x | theta_i)

For each pair, it draws posterior samples q(theta | x_i), checks whether the
true theta_i falls inside equal-tailed alpha-credible intervals, and then
aggregates the empirical coverage curve and calibration error curve.
"""

import argparse
import glob
import json
import os
import sys
import time

# Make the experiment package and the repository root importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbi_common.config_utils import cli_config_arg, resolve_config_path

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

try:
    import bayesflow  # noqa: F401

    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import keras
from torch.utils.data import DataLoader, TensorDataset

from data_generation import SIRTask, d, d_x
from models.dnnabc import DNNABC_Model, abc_rejection_sampling, train_dnnabc
from models.mmd import MMD_Model, mmd_loss
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.bayesflow_net import build_bayesflow_model
from utilities import compute_metrics, refine_posterior

# Config file selection: --config <path> > SIR_CONFIG_PATH env var >
# config.json next to this script (see sbi_common/config_utils.py).
_CONFIG_PATH, _REQUESTED_CONFIG_PATH = resolve_config_path(
    _SCRIPT_DIR, "SIR_CONFIG_PATH", cli_value=cli_config_arg()
)
try:
    with open(_CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
    print(f"Loaded configuration from {_CONFIG_PATH}")
except (FileNotFoundError, TypeError):
    if _REQUESTED_CONFIG_PATH:
        print(
            f"Warning: requested config '{_REQUESTED_CONFIG_PATH}' was not found. "
            "Using default parameters."
        )
    else:
        print("Warning: config.json not found. Using default parameters.")
    CONFIG = {
        "num_rounds": 1,
        "N": 1000000.0,
        "T_MAX": 160.0,
        "NUM_OBS": 10,
        "dataset_size": 12800,
        "val_size": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "n_samples_posterior": 1000,
        "models_to_run": ["smmd", "mmd", "bayesflow", "dnnabc"],
        "models_config": {
            "smmd": {"epochs": 300, "summary_dim": 4},
            "mmd": {"epochs": 300, "summary_dim": 4},
            "bayesflow": {"epochs": 300, "summary_dim": 4},
            "dnnabc": {"epochs": 300, "n_pool": 34400},
        },
        "smmd_mmd_config": {
            "M": 50,
            "L": 20,
            "bandwidth_n_samples": 4400,
        },
        "refinement_config": {
            "n_chains": 1000,
            "burn_in": 29,
            "n_samples_per_chain": 1,
            "proposal_std": 0.5,
        },
    }

DATASET_SIZE = CONFIG.get("dataset_size", 12800)
BATCH_SIZE = CONFIG.get("batch_size", 128)
MODELS_TO_RUN = CONFIG.get("models_to_run", ["smmd", "mmd", "bayesflow", "dnnabc"])
MODELS_CONFIG = CONFIG.get("models_config", {})

SMMD_MMD_CONFIG = CONFIG.get(
    "smmd_mmd_config", {"M": 50, "L": 20, "bandwidth_n_samples": 4400}
)
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)

REFINEMENT_CONFIG = CONFIG.get(
    "refinement_config",
    {"n_chains": 1000, "burn_in": 29, "n_samples_per_chain": 1},
)

LEARNING_RATE = CONFIG.get("learning_rate", 1e-3)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", 1000)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def get_scheduler(optimizer, epochs):
    """Return a cosine annealing learning rate scheduler used for SMMD/MMD."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)


def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", n_time_steps=50, label=""):
    """
    Train SMMD or MMD model for a given number of epochs.

    Args:
        model: SMMD_Model or MMD_Model instance.
        train_loader: DataLoader yielding (x, theta) batches.
        epochs: number of training epochs.
        device: torch.device on which to run training.
        model_type: "smmd" or "mmd" to pick the correct loss.
        n_time_steps: used only to scale the kernel bandwidth in the loss.
        label: prefix for log messages (e.g. "[R1] SMMD").

    Returns:
        loss_history: list of average loss values per epoch.
        training_time: wall-clock training time in seconds.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    model.to(device)
    model.train()
    loss_history = []
    prefix = f"[{label}] " if label else ""
    tag = model_type.upper()
    print(f"{prefix}Starting training ({tag}) for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            optimizer.zero_grad()
            with torch.enable_grad():
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                theta_fake = model(x_batch, z)
                if model_type == "smmd":
                    loss = sliced_mmd_loss(
                        theta_batch, theta_fake, num_slices=L, n_time_steps=n_time_steps
                    )
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_time_steps=n_time_steps)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"{prefix}[{tag}] Epoch {epoch+1}/{epochs}, loss = {avg_loss:.4f}")
    training_time = time.time() - start_time
    print(f"{prefix}Finished training ({tag}) in {training_time:.2f} seconds")
    return loss_history, training_time


def train_bayesflow(train_loader, epochs, _device, summary_dim=10, label=""):
    """
    Build and train a BayesFlow amortized posterior for the SIR task.

    The BayesFlow model uses Keras with a PyTorch backend. We wrap the training
    loop here so the rest of the script can treat it like a standard model.
    """
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
    prefix = f"[{label}] " if label else ""
    tag = "BAYESFLOW"
    print(f"{prefix}Starting training ({tag}) for {epochs} epochs...")
    first_batch = next(iter(train_loader))
    first_x, first_theta = first_batch[0], first_batch[1]
    amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=summary_dim)
    try:
        dummy_dict = {
            "inference_variables": first_theta.float(),
            "summary_variables": first_x.float(),
        }
        if hasattr(amortized_posterior, "adapter"):
            _ = amortized_posterior.adapter(dummy_dict)
        _ = amortized_posterior.log_prob(dummy_dict)
    except Exception:
        pass
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(LEARNING_RATE),
        decay_steps=total_steps,
        alpha=0.0,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    amortized_posterior.compile(optimizer=optimizer)
    loss_history = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch in train_loader:
            x_batch = batch[0].float()
            theta_batch = batch[1].float()
            batch_dict = {
                "inference_variables": theta_batch,
                "summary_variables": x_batch,
            }
            metrics = amortized_posterior.train_step(batch_dict)
            loss_val = metrics.get("loss", list(metrics.values())[0])
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            else:
                loss_val = float(loss_val)
            epoch_loss += loss_val
            num_batches += 1
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"{prefix}[{tag}] Epoch {epoch+1}/{epochs}, loss = {avg_loss:.4f}")
    training_time = time.time() - start_time
    print(f"{prefix}Finished training ({tag}) in {training_time:.2f} seconds")
    return amortized_posterior, loss_history, training_time


PARAM_NAMES = [r"$\beta$", r"$\gamma$"]
CAL_PAIR_SEED_OFFSET = 1_000_000  # separates calibration-pair RNG from the training RNG
REFINE_SEED_OFFSET = 500_000      # separates per-pair refinement RNG from the amortized RNG
MODEL_COLORS = {
    "SMMD": "#DB1218",
    "MMD": "#1083CA",
    "BAYESFLOW": "#49C926",
    "DNNABC": "#8315dd",
}
STAGE_LABELS = {"initial": "Amortized", "refine": "Refined"}
STAGE_LINESTYLES = {"initial": "--", "refine": "-"}


def parse_args():
    parser = argparse.ArgumentParser(description="SIR expected coverage experiment.")
    parser.add_argument(
        "--config", "--config-path", dest="config_path", type=str, default=None,
        help="Path to the experiment config (default: config.json next to this script).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "calibration_expected_coverage"
        ),
    )
    parser.add_argument("--num-calibration-pairs", type=int, default=None)
    parser.add_argument("--round-size", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-samples-posterior", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--save-round-plots", action="store_true")
    parser.add_argument("--max-plot-rounds", type=int, default=None)
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun every round even if its artifacts already exist.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def train_models(task, train_loader, models_to_run, models_config):
    trained = {}
    timings = {}

    for model_name in models_to_run:
        model_conf = models_config.get(model_name, {})
        if isinstance(model_conf, int):
            model_conf = {"epochs": model_conf}
        epochs = int(model_conf.get("epochs", 50))
        summary_dim = int(model_conf.get("summary_dim", 8))

        print("\n" + "=" * 72)
        print(f"Training model: {model_name} | epochs={epochs}")
        print("=" * 72)

        if model_name == "smmd":
            model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=task.n_obs)
            _, training_time = train_smmd_mmd(
                model, train_loader, epochs, DEVICE, "smmd", n_time_steps=task.n_obs
            )
        elif model_name == "mmd":
            model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=task.n_obs)
            _, training_time = train_smmd_mmd(
                model, train_loader, epochs, DEVICE, "mmd", n_time_steps=task.n_obs
            )
        elif model_name == "bayesflow":
            if not BAYESFLOW_AVAILABLE:
                print("Skipping BayesFlow because the package is not available.")
                continue
            model, _, training_time = train_bayesflow(
                train_loader, epochs, DEVICE, summary_dim=summary_dim
            )
        elif model_name == "dnnabc":
            model = DNNABC_Model(d=d, d_x=d_x)
            t0 = time.time()
            _ = train_dnnabc(model, train_loader, epochs, DEVICE)
            training_time = time.time() - t0
        else:
            print(f"Skipping unsupported model in calibration script: {model_name}")
            continue

        trained[model_name] = model
        timings[model_name] = float(training_time)

    return trained, timings


def sample_posterior_for_model(model_name, model, x_obs, task, n_samples, model_conf):
    if model_name in ["smmd", "mmd"]:
        model.eval()
        x_tensor = torch.from_numpy(x_obs).float().to(DEVICE)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
        with torch.no_grad():
            stats = model.T(x_tensor)
            z = torch.randn(1, n_samples, d, device=DEVICE)
            theta_samples = model.G(z, stats).squeeze(0).cpu().numpy()
        return theta_samples

    if model_name == "bayesflow":
        x_np = np.asarray(x_obs, dtype=np.float32)
        if x_np.ndim == 2:
            x_np = x_np[np.newaxis, ...]
        post = model.sample(conditions={"summary_variables": x_np}, num_samples=n_samples)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if hasattr(post, "numpy"):
            post = post.numpy()
        return np.asarray(post).reshape(-1, d)

    if model_name == "dnnabc":
        return abc_rejection_sampling(
            model,
            x_obs,
            task,
            n_samples=n_samples,
            n_pool=int(model_conf.get("n_pool", 6000)),
            device=DEVICE,
        )

    raise ValueError(f"Unsupported model for posterior sampling: {model_name}")


def supports_refinement(model_name):
    return model_name in {"smmd", "mmd", "bayesflow"}


def sample_refined_posterior_for_model(
    model_name,
    model,
    x_obs,
    task,
    refinement_config,
    smmd_mmd_config,
):
    if not supports_refinement(model_name):
        return None
    return refine_posterior(
        model,
        x_obs,
        task=task,
        n_chains=int(refinement_config.get("n_chains", 1000)),
        n_samples=int(refinement_config.get("n_samples_per_chain", 1)),
        burn_in=int(refinement_config.get("burn_in", 49)),
        thin=1,
        device=DEVICE,
        bandwidth_n_samples=int(smmd_mmd_config.get("bandwidth_n_samples", 5000)),
    )


def compute_pair_quantiles(samples, n_steps):
    """Return equal-tailed interval endpoints for each credibility level.

    ``samples`` has shape ``(n_samples, d)``; the output arrays have shape
    ``(n_steps, d)`` with rows ordered by increasing credibility.
    """
    confidences = np.linspace(0.0, 1.0, int(n_steps) + 1, endpoint=False, dtype=np.float32)[1:]
    q_values = []
    for conf in confidences:
        q_low = 0.5 * (1.0 - float(conf))
        q_high = 0.5 * (1.0 + float(conf))
        q_values.extend([q_low, q_high])
    quantiles = np.quantile(np.asarray(samples, dtype=np.float32), q_values, axis=0)
    return quantiles[0::2], quantiles[1::2]  # low, high: (n_steps, d)


def compute_expected_coverage_from_quantiles(low, high, theta_true_by_pair, n_steps=100):
    """Aggregate per-pair interval endpoints into an expected-coverage curve.

    ``low``/``high`` have shape ``(n_pairs, n_steps, d)`` and
    ``theta_true_by_pair`` has shape ``(n_pairs, d)``.  This is the memory-lean
    equivalent of computing quantiles over the raw posterior samples.
    """
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    theta_true_by_pair = np.asarray(theta_true_by_pair, dtype=np.float32)

    confidences = np.linspace(0.0, 1.0, int(n_steps) + 1, endpoint=False, dtype=np.float32)[1:]
    inliers = (theta_true_by_pair[:, None, :] > low) & (theta_true_by_pair[:, None, :] < high)
    empirical_coverage = np.mean(inliers.astype(np.float32), axis=0)
    median_interval_lengths = np.median((high - low).astype(np.float32), axis=0)

    calibration_error = empirical_coverage - confidences[:, None]
    return {
        "confidences": confidences,
        "empirical_coverage": empirical_coverage,
        "median_interval_lengths": median_interval_lengths,
        "calibration_error": calibration_error,
        "median_abs_calibration_error_per_dim": np.median(np.abs(calibration_error), axis=0).astype(np.float32),
        "median_abs_calibration_error_mean": float(np.mean(np.median(np.abs(calibration_error), axis=0))),
    }


def build_series_entries(results_by_series):
    entries = []
    for (model_name, stage), res in results_by_series.items():
        key = model_name.upper()
        stage_label = STAGE_LABELS.get(stage, stage.title())
        entries.append(
            {
                "model": model_name,
                "stage": stage,
                "label": f"{key} ({stage_label})",
                "color": MODEL_COLORS.get(key, "#333333"),
                "linestyle": STAGE_LINESTYLES.get(stage, "-"),
                "result": res,
            }
        )
    return entries


def plot_calibration_error_curves(results_by_series, param_names, output_dir):
    ensure_dir(output_dir)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    entries = build_series_entries(results_by_series)
    d_param = len(param_names)
    fig, axes = plt.subplots(1, d_param, figsize=(7 * d_param, 5.5), sharex=True, sharey=True)
    if d_param == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        for entry in entries:
            res = entry["result"]
            ax.plot(
                res["confidences"],
                res["calibration_error"][:, j],
                color=entry["color"],
                linewidth=2.8,
                linestyle=entry["linestyle"],
                label=entry["label"],
            )
        ax.axhline(0.0, color="#4d4d4d", linestyle="-", linewidth=2.2, alpha=0.85)
        ax.set_title(param_names[j], fontsize=22, fontweight="bold", pad=10)
        ax.set_xlabel(r"Confidence $q$", fontsize=17)
        if j == 0:
            ax.set_ylabel(r"Calibration error $q_{\mathrm{inliers}} - q$", fontsize=17)
        ax.tick_params(axis="both", labelsize=14, width=1.5, length=5)
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#222222")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(by_label)),
        fontsize=13,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_error_curves.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    for j, param_name in enumerate(param_names):
        fig_j, ax_j = plt.subplots(1, 1, figsize=(9, 6))
        for entry in entries:
            res = entry["result"]
            ax_j.plot(
                res["confidences"],
                res["calibration_error"][:, j],
                color=entry["color"],
                linewidth=2.8,
                linestyle=entry["linestyle"],
                label=entry["label"],
            )
        ax_j.axhline(0.0, color="#4d4d4d", linestyle="-", linewidth=2.2, alpha=0.85)
        ax_j.set_title(param_name, fontsize=24, fontweight="bold", pad=10)
        ax_j.set_xlabel(r"Confidence $q$", fontsize=18)
        ax_j.set_ylabel(r"Calibration error $q_{\mathrm{inliers}} - q$", fontsize=18)
        ax_j.tick_params(axis="both", labelsize=14, width=1.5, length=5)
        ax_j.grid(True, linestyle="--", alpha=0.30, linewidth=0.9)
        for spine in ax_j.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#222222")
        ax_j.legend(loc="upper left", fontsize=13, frameon=True)
        fig_j.tight_layout()
        fig_j.savefig(
            os.path.join(output_dir, f"calibration_error_curve_param{j+1}.png"),
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig_j)


def plot_round_posterior_comparison(theta_true, sample_entries, output_path):
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(14, 5.5))
    if len(PARAM_NAMES) == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        for entry in sample_entries:
            sns.kdeplot(
                x=np.asarray(entry["samples"])[:, j],
                ax=ax,
                color=entry["color"],
                linewidth=2.6,
                linestyle=entry["linestyle"],
                fill=False,
                label=entry["label"],
            )
        ax.axvline(theta_true[j], color="#222222", linestyle="-", linewidth=2.2, alpha=0.9, label="True Value")
        ax.set_title(PARAM_NAMES[j], fontsize=22, fontweight="bold", pad=10)
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("Density", fontsize=17)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=14, width=1.5, length=5)
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#222222")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(by_label)),
        fontsize=13,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_round_check_artifacts(output_dir, round_idx, theta_true, sample_entries, metrics_by_model):
    round_dir = ensure_dir(os.path.join(output_dir, "round_checks", f"round_{round_idx:04d}"))
    npz_path = os.path.join(round_dir, "posterior_samples.npz")

    # Merge with whatever this round already has on disk so that resumed runs
    # keep stages finished in earlier runs (and do not overwrite them away).
    stored = {}
    if os.path.exists(npz_path):
        with np.load(npz_path) as z:
            stored = {key: z[key] for key in z.files}
    stored["theta_true"] = np.asarray(theta_true, dtype=np.float32)
    for entry in sample_entries:
        stored[f"{entry['model']}_{entry['stage']}"] = np.asarray(entry["samples"], dtype=np.float32)

    # Rebuild the full entry list: this run's entries first, then any stages
    # that only exist from an earlier run (sorted for a stable legend order).
    merged_entries = list(sample_entries)
    seen = {(e["model"], e["stage"]) for e in merged_entries}
    for key in sorted(stored):
        if key == "theta_true":
            continue
        model_name, stage = key.rsplit("_", 1)
        if (model_name, stage) in seen:
            continue
        merged_entries.append(
            {
                "model": model_name,
                "stage": stage,
                "label": f"{model_name.upper()} ({STAGE_LABELS.get(stage, stage.title())})",
                "color": MODEL_COLORS.get(model_name.upper(), "#333333"),
                "linestyle": STAGE_LINESTYLES.get(stage, "-"),
                "samples": stored[key],
            }
        )
        seen.add((model_name, stage))

    np.savez(npz_path, **stored)
    plot_round_posterior_comparison(
        theta_true=theta_true,
        sample_entries=merged_entries,
        output_path=os.path.join(round_dir, "posterior_comparison.png"),
    )
    # Merge metric rows too, replacing rows of stages that were re-run.
    metrics_path = os.path.join(round_dir, "posterior_metrics.csv")
    metrics_df = pd.DataFrame(metrics_by_model)
    if os.path.exists(metrics_path):
        metrics_df = pd.concat(
            [pd.read_csv(metrics_path), metrics_df], ignore_index=True
        ).drop_duplicates(subset=["model", "stage", "round"], keep="last")
    metrics_df.to_csv(metrics_path, index=False)
    with open(os.path.join(round_dir, "theta_true.json"), "w") as f:
        json.dump({"theta_true": np.asarray(theta_true, dtype=float).tolist()}, f, indent=2)


def load_or_sample_calibration_pairs(task, num_pairs, cal_seed, output_dir):
    """Load cached (theta, x) calibration pairs or draw them with a dedicated RNG.

    The pairs are drawn from the joint distribution ``p(theta, x)`` with an
    RNG seeded by ``cal_seed``, which is decoupled from the training RNG, and
    cached on disk so that reruns (and resume runs) use exactly the same pairs.
    """
    pairs_dir = ensure_dir(os.path.join(output_dir, "calibration_pairs"))
    theta_path = os.path.join(pairs_dir, "theta_cal.npy")
    x_path = os.path.join(pairs_dir, "x_cal.npy")
    meta_path = os.path.join(pairs_dir, "meta.json")

    if os.path.exists(theta_path) and os.path.exists(x_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("cal_seed") == cal_seed and meta.get("num_pairs") == num_pairs:
            print(f"Loaded {num_pairs} cached calibration pairs (cal_seed={cal_seed}).")
            return np.load(theta_path), np.load(x_path)

    np.random.seed(cal_seed)
    theta_cal = task.sample_prior(num_pairs).astype(np.float32)
    x_cal = task.simulator(theta_cal).astype(np.float32)
    np.save(theta_path, theta_cal)
    np.save(x_path, x_cal)
    with open(meta_path, "w") as f:
        json.dump({"cal_seed": int(cal_seed), "num_pairs": int(num_pairs)}, f, indent=2)
    print(f"Sampled {num_pairs} calibration pairs with dedicated cal_seed={cal_seed}.")
    return theta_cal, x_cal


def _round_paths(output_dir, round_idx):
    """Return (per-pair csv, quantiles npz) paths for one round."""
    return (
        os.path.join(output_dir, f"per_pair_round_{round_idx:04d}.csv"),
        os.path.join(output_dir, f"quantiles_round_{round_idx:04d}.npz"),
    )


def stage_done(output_dir, round_idx, model_name, stage):
    """Whether a (model, stage, round) already has quantile artifacts on disk."""
    _, npz_path = _round_paths(output_dir, round_idx)
    if not os.path.exists(npz_path):
        return False
    with np.load(npz_path) as z:
        return f"{model_name}_{stage}_low" in z and f"{model_name}_{stage}_high" in z


def save_round_artifacts(output_dir, round_idx, model_name, stage, rows, low, high, theta_true_slice):
    """Persist one (model, stage, round): append per-pair rows and merge quantiles."""
    csv_path, npz_path = _round_paths(output_dir, round_idx)

    df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    data = {}
    if os.path.exists(npz_path):
        with np.load(npz_path) as z:
            data = {key: z[key] for key in z.files}
    data["theta_true"] = np.asarray(theta_true_slice, dtype=np.float32)
    data[f"{model_name}_{stage}_low"] = np.asarray(low, dtype=np.float32)
    data[f"{model_name}_{stage}_high"] = np.asarray(high, dtype=np.float32)
    np.savez(npz_path, **data)


def load_accumulated(output_dir):
    """Load all round artifacts; return (per_pair_df, low, high, theta_true).

    ``low``/``high`` are dicts keyed by ``"{model}_{stage}"`` holding arrays of
    shape ``(n_pairs_done, n_steps, d)`` concatenated in round order.
    """
    frames = []
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "per_pair_round_*.csv"))):
        frames.append(pd.read_csv(csv_path))
    per_pair = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    npz_paths = sorted(glob.glob(os.path.join(output_dir, "quantiles_round_*.npz")))
    keys = set()
    theta_parts = []
    for path in npz_paths:
        with np.load(path) as z:
            for key in z.files:
                if key.endswith(("_low", "_high")):
                    keys.add(key.rsplit("_", 1)[0])
            if "theta_true" in z:
                theta_parts.append(z["theta_true"])

    low, high = {}, {}
    for key in sorted(keys):
        low_parts, high_parts = [], []
        for path in npz_paths:
            with np.load(path) as z:
                if f"{key}_low" in z:
                    low_parts.append(z[f"{key}_low"])
                    high_parts.append(z[f"{key}_high"])
        low[key] = np.concatenate(low_parts, axis=0) if low_parts else np.empty((0,), dtype=np.float32)
        high[key] = np.concatenate(high_parts, axis=0) if high_parts else np.empty((0,), dtype=np.float32)

    theta_true = np.concatenate(theta_parts, axis=0) if theta_parts else np.empty((0,), dtype=np.float32)
    return per_pair, low, high, theta_true


def build_summary_row(model_name, stage, sub, res, training_time, round_idx):
    """Build one cumulative summary row from the per-pair metrics of a stage."""
    return {
        "round": int(round_idx),
        "model": model_name,
        "stage": stage,
        "num_pairs": int(len(sub)),
        "training_time_sec": float(training_time),
        "calibration_eval_time_sec": float(sub["seconds"].sum()),
        "avg_time_sec": float(sub["seconds"].mean()),
        "bias_mean": float(sub["bias_l2"].mean()),
        "bias_median": float(sub["bias_l2"].median()),
        "bias_std": float(sub["bias_l2"].std(ddof=0)),
        "rmse_mean": float(sub["rmse"].mean()),
        "rmse_std": float(sub["rmse"].std(ddof=0)),
        "hdi_len_param1_mean": float(sub["hdi_length_param1"].mean()),
        "hdi_len_param2_mean": float(sub["hdi_length_param2"].mean()),
        "coverage_param1_mean": float(sub["coverage_param1"].mean()),
        "coverage_param2_mean": float(sub["coverage_param2"].mean()),
        "median_abs_calibration_error_mean": float(res["median_abs_calibration_error_mean"]),
        "median_abs_calibration_error_param1": float(res["median_abs_calibration_error_per_dim"][0]),
        "median_abs_calibration_error_param2": float(res["median_abs_calibration_error_per_dim"][1]),
    }


def update_summary_and_curves(output_dir, n_steps, training_times, round_idx, record_round=True):
    """Recompute cumulative summaries, curve data, and plots from all round artifacts.

    Called after every completed round so that intermediate results are
    visible while the experiment is still running.
    """
    per_pair, low, high, theta_true = load_accumulated(output_dir)
    results_by_series = {}
    summary_rows = []

    for key in sorted(low):
        if key.endswith("_refine"):
            model_name, stage = key[:-7], "refine"
        elif key.endswith("_initial"):
            model_name, stage = key[:-8], "initial"
        else:
            continue
        n_pairs_done = low[key].shape[0]
        res = compute_expected_coverage_from_quantiles(
            low[key], high[key], theta_true[:n_pairs_done], n_steps=n_steps
        )
        results_by_series[(model_name, stage)] = res
        sub = per_pair[(per_pair["model"] == model_name) & (per_pair["stage"] == stage)]
        summary_rows.append(build_summary_row(
            model_name, stage, sub, res, training_times.get(model_name, np.nan), round_idx
        ))

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(os.path.join(output_dir, "calibration_summary.csv"), index=False)

        if record_round:
            by_round_path = os.path.join(output_dir, "calibration_summary_by_round.csv")
            by_round = summary.copy()
            if os.path.exists(by_round_path):
                by_round = pd.concat([pd.read_csv(by_round_path), by_round], ignore_index=True)
            by_round.to_csv(by_round_path, index=False)

    coverage_rows = []
    for (model_name, stage), res in results_by_series.items():
        for step_idx, q in enumerate(res["confidences"]):
            for param_idx, param_name in enumerate(PARAM_NAMES):
                coverage_rows.append(
                    {
                        "model": model_name,
                        "stage": stage,
                        "param": param_name,
                        "confidence": float(q),
                        "empirical_coverage": float(res["empirical_coverage"][step_idx, param_idx]),
                        "calibration_error": float(res["calibration_error"][step_idx, param_idx]),
                        "abs_calibration_error": float(abs(res["calibration_error"][step_idx, param_idx])),
                        "median_interval_length": float(res["median_interval_lengths"][step_idx, param_idx]),
                    }
                )
    pd.DataFrame(coverage_rows).to_csv(os.path.join(output_dir, "calibration_curve_data.csv"), index=False)

    if results_by_series:
        plot_calibration_error_curves(results_by_series, PARAM_NAMES, output_dir)


def main():
    args = parse_args()
    config_path, _requested_cfg = resolve_config_path(
        _SCRIPT_DIR, "SIR_CONFIG_PATH", cli_value=args.config_path
    )
    if config_path is None:
        raise FileNotFoundError(
            f"Config file not found (requested: {_requested_cfg or 'config.json'})."
        )
    config = load_config(config_path)
    calibration_config = config.get("calibration_config", {})

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = ensure_dir(args.output_dir)

    task = SIRTask(seed=seed)
    dataset_size = int(args.dataset_size or config.get("dataset_size", DATASET_SIZE))
    batch_size = int(args.batch_size or config.get("batch_size", BATCH_SIZE))
    n_samples_posterior = int(args.n_samples_posterior or config.get("n_samples_posterior", N_SAMPLES_POSTERIOR))
    num_calibration_pairs = int(
        args.num_calibration_pairs
        if args.num_calibration_pairs is not None
        else calibration_config.get("num_calibration_pairs", 100)
    )
    round_size = int(
        args.round_size
        if args.round_size is not None
        else calibration_config.get("round_size", 100)
    )
    n_steps = int(
        args.n_steps
        if args.n_steps is not None
        else calibration_config.get("n_steps", 100)
    )
    save_round_plots = bool(
        args.save_round_plots or calibration_config.get("save_round_plots", False)
    )
    max_plot_rounds = int(
        args.max_plot_rounds
        if args.max_plot_rounds is not None
        else calibration_config.get("max_plot_rounds", 10)
    )
    models_to_run = args.models or config.get("models_to_run", MODELS_TO_RUN)
    models_config = config.get("models_config", MODELS_CONFIG)
    smmd_mmd_config = config.get("smmd_mmd_config", SMMD_MMD_CONFIG)
    refinement_config = config.get("refinement_config", REFINEMENT_CONFIG)
    resume = not args.no_resume

    print("=" * 80)
    print("SIR expected coverage experiment")
    print(f"Using device: {DEVICE}")
    print(f"Training dataset size: {dataset_size}")
    print(f"Calibration pairs: {num_calibration_pairs} (round size: {round_size})")
    print(f"Calibration steps: {n_steps}")
    print(f"Seed: {seed} (calibration pairs use seed+{CAL_PAIR_SEED_OFFSET})")
    print(f"Resume enabled: {resume}")
    print(f"Save round plots: {save_round_plots}")
    print(f"Max plot rounds: {max_plot_rounds}")
    print(f"Models: {models_to_run}")
    print("=" * 80)

    # 1. Train every amortized model once (reproducible with `seed`; Keras
    #    models need their own global seed on top of the numpy/torch seeds).
    np.random.seed(seed)
    torch.manual_seed(seed)
    keras.utils.set_random_seed(seed)
    theta_train = task.sample_prior(dataset_size)
    x_train = task.simulator(theta_train)
    dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(theta_train).float())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trained_models, training_times = train_models(
        task=task,
        train_loader=train_loader,
        models_to_run=models_to_run,
        models_config=models_config,
    )

    # 2. Draw (or load cached) calibration pairs from p(theta, x) with a
    #    dedicated RNG so they are decoupled from the training RNG.
    cal_seed = seed + CAL_PAIR_SEED_OFFSET
    task_cal = SIRTask(seed=cal_seed)
    theta_cal, x_cal = load_or_sample_calibration_pairs(
        task_cal, num_calibration_pairs, cal_seed, output_dir
    )

    n_rounds = int(np.ceil(num_calibration_pairs / round_size))
    round_plot_cache = {
        pair_idx: [] for pair_idx in range(1, min(num_calibration_pairs, max_plot_rounds) + 1)
    }
    round_plot_metrics_cache = {
        pair_idx: [] for pair_idx in range(1, min(num_calibration_pairs, max_plot_rounds) + 1)
    }

    # 3. Per-round calibration evaluation; summaries and plots are refreshed
    #    after every completed round so progress is visible while running.
    for round_idx in range(1, n_rounds + 1):
        idx0 = (round_idx - 1) * round_size
        idx1 = min(idx0 + round_size, num_calibration_pairs)
        for model_name, model in trained_models.items():
            print("\n" + "-" * 72)
            print(f"Calibration evaluation for {model_name} (round {round_idx}/{n_rounds})")
            print("-" * 72)
            model_conf = models_config.get(model_name, {})
            stages = [("initial", False)]
            if supports_refinement(model_name) and not args.skip_refinement:
                stages.append(("refine", True))

            for stage, do_refine in stages:
                if resume and stage_done(output_dir, round_idx, model_name, stage):
                    print(f"  {model_name}/{stage}: round {round_idx} already on disk, skipping.")
                    continue

                rows = []
                low_parts = []
                high_parts = []
                for idx in range(idx0, idx1):
                    pair_idx = idx + 1
                    # Per-pair, per-stage seed: amortized and refined draws stay
                    # reproducible even when a single stage is resumed alone.
                    stage_seed = cal_seed + pair_idx + (REFINE_SEED_OFFSET if do_refine else 0)
                    np.random.seed(stage_seed)
                    torch.manual_seed(stage_seed)
                    pair_task = SIRTask(seed=stage_seed)

                    t_pair = time.time()
                    if do_refine:
                        samples = sample_refined_posterior_for_model(
                            model_name=model_name,
                            model=model,
                            x_obs=x_cal[idx],
                            task=pair_task,
                            refinement_config=refinement_config,
                            smmd_mmd_config=smmd_mmd_config,
                        )
                    else:
                        samples = sample_posterior_for_model(
                            model_name=model_name,
                            model=model,
                            x_obs=x_cal[idx],
                            task=pair_task,
                            n_samples=n_samples_posterior,
                            model_conf=model_conf,
                        )
                    elapsed_pair = time.time() - t_pair
                    samples = np.asarray(samples, dtype=np.float32)
                    metric = compute_metrics(samples, theta_cal[idx])
                    low, high = compute_pair_quantiles(samples, n_steps)

                    rows.append(
                        {
                            "model": model_name,
                            "stage": stage,
                            "pair_idx": int(pair_idx),
                            "round": int(round_idx),
                            "theta_true_param1": float(theta_cal[idx][0]),
                            "theta_true_param2": float(theta_cal[idx][1]),
                            "bias_l2": float(metric["bias_l2"]),
                            "rmse": float(metric["rmse"]),
                            "hdi_length_param1": float(metric["hdi_length"][0]),
                            "hdi_length_param2": float(metric["hdi_length"][1]),
                            "coverage_param1": float(metric["coverage"][0]),
                            "coverage_param2": float(metric["coverage"][1]),
                            "seconds": float(elapsed_pair),
                        }
                    )
                    low_parts.append(low)
                    high_parts.append(high)

                    if save_round_plots and pair_idx <= max_plot_rounds:
                        round_plot_cache[pair_idx].append(
                            {
                                "model": model_name,
                                "stage": stage,
                                "label": f"{model_name.upper()} ({STAGE_LABELS[stage]})",
                                "color": MODEL_COLORS.get(model_name.upper(), "#333333"),
                                "linestyle": STAGE_LINESTYLES[stage],
                                "samples": samples,
                            }
                        )
                        round_plot_metrics_cache[pair_idx].append(
                            {
                                "model": model_name,
                                "stage": stage,
                                "round": int(pair_idx),
                                "bias_l2": float(metric["bias_l2"]),
                                "rmse": float(metric["rmse"]),
                                "hdi_length_param1": float(metric["hdi_length"][0]),
                                "hdi_length_param2": float(metric["hdi_length"][1]),
                                "coverage_param1": float(metric["coverage"][0]),
                                "coverage_param2": float(metric["coverage"][1]),
                            }
                        )
                        # Refresh this pair's posterior-comparison plot right
                        # away so progress is visible model by model.
                        save_round_check_artifacts(
                            output_dir=output_dir,
                            round_idx=pair_idx,
                            theta_true=theta_cal[idx],
                            sample_entries=round_plot_cache[pair_idx],
                            metrics_by_model=round_plot_metrics_cache[pair_idx],
                        )

                low_arr = np.stack(low_parts, axis=0)
                high_arr = np.stack(high_parts, axis=0)
                save_round_artifacts(
                    output_dir, round_idx, model_name, stage, rows,
                    low_arr, high_arr, theta_cal[idx0:idx1],
                )
                print(f"  {model_name}/{stage}: round {round_idx}/{n_rounds} done ({len(rows)} pairs).")

        # 4. Refresh cumulative summaries and plots for this round.
        update_summary_and_curves(output_dir, n_steps, training_times, round_idx)
        print(f"ROUND {round_idx}/{n_rounds} COMPLETE: summary and calibration curves updated.")

    # 5. Final refresh (also covers resume runs that skipped every round);
    #    do not append a duplicate round record to the by-round table.
    update_summary_and_curves(output_dir, n_steps, training_times, n_rounds, record_round=False)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "seed": int(seed),
                "cal_seed": int(cal_seed),
                "num_calibration_pairs": num_calibration_pairs,
                "round_size": int(round_size),
                "n_rounds": int(n_rounds),
                "n_steps": n_steps,
                "dataset_size": int(dataset_size),
                "batch_size": int(batch_size),
                "n_samples_posterior": int(n_samples_posterior),
                "models": list(trained_models.keys()),
                "device": str(DEVICE),
                "save_round_plots": save_round_plots,
                "max_plot_rounds": max_plot_rounds,
                "skip_refinement": bool(args.skip_refinement),
                "calibration_config": calibration_config,
                "refinement_config": refinement_config,
                "smmd_mmd_config": smmd_mmd_config,
            },
            f,
            indent=2,
        )

    print("\nSaved outputs:")
    print(f"  per-round pair metrics: {output_dir}/per_pair_round_*.csv")
    print(f"  per-round quantiles: {output_dir}/quantiles_round_*.npz")
    print(f"  cumulative summary: {os.path.join(output_dir, 'calibration_summary.csv')}")
    print(f"  summary by round: {os.path.join(output_dir, 'calibration_summary_by_round.csv')}")
    print(f"  curve data: {os.path.join(output_dir, 'calibration_curve_data.csv')}")
    print(f"  error plot: {os.path.join(output_dir, 'calibration_error_curves.png')}")
    if save_round_plots:
        print(f"  round checks: {os.path.join(output_dir, 'round_checks')}")
        print(f"  round posterior samples: {os.path.join(output_dir, 'round_checks', 'round_*', 'posterior_samples.npz')}")

if __name__ == "__main__":
    main()
