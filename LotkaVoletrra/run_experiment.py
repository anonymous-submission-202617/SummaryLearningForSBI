###############################################################################
# Lotka-Volterra model with vague prior
#
# Trains SMMD / MMD / BayesFlow (Refine+: sequential training + ABC-MCMC
# refinement) and the DNN-ABC / SNPE-C baselines over 100 replications at a
# fixed true parameter, under a matched simulation budget.
#
# Reproduces Table 3 ("Results for Lotka-Volterra model.") and the Refine+
# figure showing posterior densities across the amortized, sequential and
# refinement stages.
#
# Outputs: results/ (see README.md).
###############################################################################

import os
import sys
import time
from datetime import timedelta
import json
import numpy as np

# Make the experiment package and the repository root importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sbi_common.config_utils import cli_config_arg, resolve_config_path

# Reduce cross-device numeric drift on NVIDIA CUDA for ablation runs.
# Data/model randomness is fixed below from the configurable ``seed``.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

# Fix for MPS 'aten::linalg_qr.out' not implemented error
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ensure the backend is set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import keras

# Import from local modules
from data_generation import LVTask
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.bayesflow_net import build_bayesflow_model, box_to_unconstrained, unconstrained_to_box
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.sbi_wrappers import run_sbi_model
from models.theta_transforms import scale_theta_np
from utilities import (
    BoundaryCorrectedKDE,
    calibration_error_from_rounds,
    compute_metrics,
    refine_posterior,
)

# Try importing BayesFlow (optional)
try:
    import bayesflow as bf
    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False
    print("BayesFlow not installed or import failed. BayesFlow model will be unavailable.")

# ============================================================================
# 1. Configuration & Device
# ============================================================================

# Config file selection: --config <path> > positional <path> (legacy) >
# LV_CONFIG_PATH env var > config.json next to this script.
_argv = sys.argv[1:]
_cli_value = cli_config_arg(_argv)
if _cli_value is None:
    _positional = [a for a in _argv if not a.startswith("-")]
    if _positional:
        _cli_value = _positional[0]
CONFIG_PATH, _REQUESTED_CONFIG_PATH = resolve_config_path(
    _SCRIPT_DIR, "LV_CONFIG_PATH", cli_value=_cli_value
)

# Load Config
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
    print(f"✅ Loaded configuration from {CONFIG_PATH}")
except (FileNotFoundError, TypeError):
    if _REQUESTED_CONFIG_PATH:
        print(f"⚠️ Requested config '{_REQUESTED_CONFIG_PATH}' not found. Using default configuration.")
    else:
        print("⚠️ config.json not found. Using default configuration.")

    CONFIG = {
        "dataset_size": 20000,
        "val_size": 1000,
        "batch_size": 128,
        "num_rounds": 5,
        "models_to_run": ["smmd", "mmd", "bayesflow", "dnnabc"],
        "models_config": {},
        "n_observation": 151,
        "refinement_config": {
            "proposal_std_coeff": 0.5,
            "bandwidth_quantile_level": 0.01,
            "n_sims": 1
        },
        "plot_config": {
            "save_refinement_comparison_plots": False
        }
    }

# Results are written next to the script (cwd-independent), or to the
# directory requested via LV_RESULTS_DIR (used e.g. for smoke tests).
RESULTS_BASE = os.path.join(_SCRIPT_DIR, os.environ.get("LV_RESULTS_DIR", "results"))

# Reproducibility: base seed; each round is re-seeded with
# ``BASE_SEED + round_idx - 1`` so every round's data generation is
# deterministic and independently reproducible (same pattern as Radio).
BASE_SEED = int(CONFIG.get("seed", 0))
SAVE_MODEL_PARAMS = bool(CONFIG.get("save_model_params", False))

if os.environ.get("LV_QUICK_TEST") == "1":
    CONFIG = dict(CONFIG)
    CONFIG["num_rounds"] = 1
    CONFIG["dataset_size"] = min(int(CONFIG.get("dataset_size", 20000)), 512)
    CONFIG["val_size"] = min(int(CONFIG.get("val_size", 1000)), 50)
    CONFIG["n_samples_posterior"] = min(int(CONFIG.get("n_samples_posterior", 1000)), 50)

    smmd_mmd_cfg = dict(CONFIG.get("smmd_mmd_config", {}))
    smmd_mmd_cfg["initial_train_samples"] = min(int(smmd_mmd_cfg.get("initial_train_samples", 512)), 512)
    smmd_mmd_cfg["refine_train_samples"] = min(int(smmd_mmd_cfg.get("refine_train_samples", 512)), 512)
    smmd_mmd_cfg["bandwidth_n_samples"] = min(int(smmd_mmd_cfg.get("bandwidth_n_samples", 4400)), 1000)
    CONFIG["smmd_mmd_config"] = smmd_mmd_cfg

    models_cfg = dict(CONFIG.get("models_config", {}))
    for k in ["smmd", "mmd", "bayesflow"]:
        sub = dict(models_cfg.get(k, {}))
        sub["epochs"] = min(int(sub.get("epochs", 30)), 20)
        sub["num_refine_rounds"] = 1
        sub["refined_mode"] = 1
        sub["initial_train_samples"] = min(
            int(sub.get("initial_train_samples", smmd_mmd_cfg.get("initial_train_samples", 512))),
            512,
        )
        sub["refine_train_samples"] = min(
            int(sub.get("refine_train_samples", smmd_mmd_cfg.get("refine_train_samples", 512))),
            512,
        )
        sub.setdefault("mcmc_burn_in", 5)
        models_cfg[k] = sub
    dnn = dict(models_cfg.get("dnnabc", {}))
    dnn["epochs"] = min(int(dnn.get("epochs", 30)), 20)
    dnn["train_samples"] = min(int(dnn.get("train_samples", 512)), 512)
    dnn["n_pool"] = min(int(dnn.get("n_pool", 6000)), 2000)
    models_cfg["dnnabc"] = dnn
    npe = dict(models_cfg.get("npe", {}))
    npe["sbi_rounds"] = min(int(npe.get("sbi_rounds", 2)), 2)
    npe["sims_per_round"] = min(int(npe.get("sims_per_round", 1000)), 1000)
    npe["epochs"] = min(int(npe.get("epochs", 50)), 50)
    models_cfg["npe"] = npe
    CONFIG["models_config"] = models_cfg

# Hyperparameters from Config
DATASET_SIZE = CONFIG.get("dataset_size", 20000)
VAL_SIZE = CONFIG.get("val_size", 1000)
BATCH_SIZE = CONFIG.get("batch_size", 128)
NUM_ROUNDS = CONFIG.get("num_rounds", 5)
MODELS_TO_RUN = CONFIG.get("models_to_run", ["smmd"])
MODELS_CONFIG = CONFIG.get("models_config", {})
REFINEMENT_CONFIG = CONFIG.get("refinement_config", {})
PLOT_CONFIG = CONFIG.get("plot_config", {})
SAVE_REFINEMENT_COMPARISON_PLOTS = PLOT_CONFIG.get("save_refinement_comparison_plots", False)
OBSERVATION_PREPROCESSING = CONFIG.get("observation_preprocessing", {})
OBS_TRANSFORM = str(OBSERVATION_PREPROCESSING.get("transform", "none")).lower()

# SMMD/MMD specific config
SMMD_MMD_CONFIG = CONFIG.get("smmd_mmd_config", {"M": 50, "L": 20})
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)
SMMD_INITIAL_TRAIN_SAMPLES = SMMD_MMD_CONFIG.get("initial_train_samples", DATASET_SIZE)
SMMD_REFINE_TRAIN_SAMPLES = SMMD_MMD_CONFIG.get("refine_train_samples", DATASET_SIZE)
BANDWIDTH_N_SAMPLES = SMMD_MMD_CONFIG.get("bandwidth_n_samples", 4400)
MCMC_PROPOSAL_STD = REFINEMENT_CONFIG.get("proposal_std_coeff", 0.5)
BANDWIDTH_QUANTILE_LEVEL = REFINEMENT_CONFIG.get("bandwidth_quantile_level", 0.01)
N_SIMS = int(REFINEMENT_CONFIG.get("n_sims", 1))
MCMC_TOTAL_STEPS = int(REFINEMENT_CONFIG.get("mcmc_total_steps", 30))

N_TIME_STEPS = CONFIG.get("n_time_steps", 151)
DT = CONFIG.get("dt", 0.2)

LEARNING_RATE = CONFIG.get("learning_rate", 3e-4)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", 1000)

# Task Dimensions
d = 4 # alpha, beta, gamma, delta
d_x = 2 # Prey, Predator
n_obs = N_TIME_STEPS 

# Check device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✅ Using MPS (Apple Silicon) acceleration. Device: {DEVICE}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA acceleration. Device: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"ℹ️ Using CPU. Device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    print("ℹ️ CUDA determinism enabled; TF32 disabled.")

def preprocess_observations(x):
    if OBS_TRANSFORM in ("none", "", "identity"):
        return np.asarray(x, dtype=np.float32)
    x_np = np.asarray(x, dtype=np.float32)
    if OBS_TRANSFORM == "log1p":
        return np.log1p(np.clip(x_np, a_min=0.0, a_max=None)).astype(np.float32)
    raise ValueError(f"Unknown observation preprocessing transform: {OBS_TRANSFORM}")

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

# ============================================================================
# 2. Training Loops
# ============================================================================

# L1 Penalty Factor
L1_LAMBDA = 1e-4

def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", n_time_steps=151):
    """Training loop for SMMD and MMD models."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    loss_history = []
    
    print(f"Starting training ({model_type})...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            theta_batch_scaled = model.scale_theta(theta_batch)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Generate fake parameters
                # Sample Z: (batch, M, d)
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward pass
                theta_fake = model(x_batch, z)
                
                if model_type == "smmd":
                    loss = sliced_mmd_loss(theta_batch_scaled, theta_fake, num_slices=L, n_time_steps=n_time_steps)
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch_scaled, theta_fake, n_time_steps=n_time_steps)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Add L1 Penalty to Generator weights only
                if hasattr(model, 'G'):
                    l1_loss = 0.0
                    for param in model.G.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss = loss + L1_LAMBDA * l1_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    return loss_history, training_time

def train_bayesflow(
    train_loader,
    epochs,
    device,
    summary_dim=10,
    theta_lower=None,
    theta_upper=None,
    model=None,
    l1_lambda=0.0,
    weight_decay=0.0,
    learning_rate=None,
):
    """Train BayesFlow model using Keras (AdamW + per-step cosine decay)."""
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")

    print("Starting training (BayesFlow)...")

    base_lr = float(LEARNING_RATE if learning_rate is None else learning_rate)
    
    # Get n_obs from first batch
    first_batch = next(iter(train_loader))
    first_x, first_theta = first_batch[0], first_batch[1]
    
    n_current = first_x.shape[1]
    
    # Build model
    if model is None:
        amortized_posterior = build_bayesflow_model(
            d,
            d_x,
            summary_dim=summary_dim,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            inference_l1_lambda=l1_lambda,
        )
    else:
        amortized_posterior = model
        amortized_posterior.inference_l1_lambda = float(l1_lambda)
    
    # Manual build
    try:
        dummy_dict = {
            "inference_variables": torch.from_numpy(
                box_to_unconstrained(first_theta.float().cpu().numpy(), theta_lower, theta_upper)
            ).float(),
            "summary_variables": first_x.float()
        }
        if hasattr(amortized_posterior, "adapter"):
             _ = amortized_posterior.adapter(dummy_dict)
        _ = amortized_posterior.log_prob(dummy_dict)
    except Exception as e:
        print(f"Manual build warning: {e}")
        
    # AdamW + per-step cosine decay (matches the SMMD/MMD and Gaussian setups).
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(base_lr),
        decay_steps=total_steps,
        alpha=0.0,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=float(weight_decay)
    )
    amortized_posterior.compile(optimizer=optimizer)

    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            x_batch = batch[0].float()
            theta_batch = batch[1].float()
            
            # BayesFlow usually doesn't support weighted loss easily in standard train_step
            # If weights are present (len(batch) > 2), they are ignored here unless we use resampling strategy before training
            # or implement manual training loop.
            # For this implementation, we assume weighted training for BayesFlow is handled via Resampling 
            # (which happens before creating the loader).
            
            x_batch_cpu = x_batch
            theta_batch_cpu = theta_batch
            
            batch_dict = {
                "inference_variables": torch.from_numpy(
                    box_to_unconstrained(theta_batch_cpu.cpu().numpy(), theta_lower, theta_upper)
                ).float(),
                "summary_variables": x_batch_cpu
            }
            
            try:
                metrics = amortized_posterior.train_step(batch_dict)
                loss_val = metrics.get("loss", list(metrics.values())[0])
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                else:
                    loss_val = float(loss_val)
                epoch_loss += loss_val
                num_batches += 1
            except Exception as e:
                print(f"Error in train_step: {e}")
                break
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    return amortized_posterior, loss_history, training_time

def _ensure_numpy_samples(samples):
    if samples is None:
        return None
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {samples.shape}")
    return samples

def _convert_samples_for_plot(samples, sample_space="log", target_space="log"):
    samples = _ensure_numpy_samples(samples)
    if target_space == "log":
        if sample_space == "log":
            return samples
        if sample_space == "theta":
            return np.log(np.clip(samples, 1e-12, None))
    elif target_space == "theta":
        if sample_space == "log":
            return np.exp(np.clip(samples, -50.0, 50.0))
        if sample_space == "theta":
            return np.clip(samples, 1e-12, None)
    raise ValueError(f"Unsupported conversion: {sample_space} -> {target_space}")

def _resolve_sample_space(sample_spaces, key, default="log"):
    if sample_spaces is None:
        return default
    return sample_spaces.get(key, default)

def plot_posterior(theta_samples, theta_true, output_dir, filename="posterior.png", sample_space="log", plot_space="log"):
    """Plots marginal posteriors."""
    os.makedirs(output_dir, exist_ok=True)
    theta_samples = _convert_samples_for_plot(theta_samples, sample_space=sample_space, target_space=plot_space)
    theta_true = np.asarray(theta_true, dtype=float)
    theta_true_plot = theta_true if plot_space == "log" else np.exp(np.clip(theta_true, -50.0, 50.0))
    num_params = theta_samples.shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1: axes = [axes]
    
    df = pd.DataFrame(theta_samples, columns=[f'theta{i+1}' for i in range(num_params)])
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        param_symbol = rf'$\log(\theta_{{{i+1}}})$' if plot_space == "log" else rf'$\theta_{{{i+1}}}$'
        sns.kdeplot(data=df, x=param_name, fill=True, alpha=0.5, ax=ax, label='Posterior')
        if i < len(theta_true_plot):
            ax.axvline(x=theta_true_plot[i], color='red', linestyle='--', linewidth=2, label='True')
        ax.set_title(f'Marginal {param_symbol}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def format_time(seconds):
    """Formats seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def _compute_adaptive_xlim_per_dim(sample_groups, true_values=None, padding_ratio=0.12, min_padding=0.05):
    """Compute per-dimension x-axis limits from current samples and true parameters."""
    valid_groups = []
    for samples in sample_groups:
        if samples is None:
            continue
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.size == 0:
            continue
        valid_groups.append(samples)

    if not valid_groups:
        return []

    n_dims = valid_groups[0].shape[1]
    true_arr = None if true_values is None else np.asarray(true_values, dtype=float).reshape(-1)
    x_limits = []

    for dim_idx in range(n_dims):
        dim_values = []
        for samples in valid_groups:
            vals = samples[:, dim_idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                dim_values.append(vals)

        if true_arr is not None and dim_idx < true_arr.size and np.isfinite(true_arr[dim_idx]):
            dim_values.append(np.asarray([true_arr[dim_idx]], dtype=float))

        if not dim_values:
            x_limits.append(None)
            continue

        merged = np.concatenate(dim_values)
        x_min = float(np.min(merged))
        x_max = float(np.max(merged))
        span = x_max - x_min

        if span <= 1e-8:
            center_scale = max(abs(x_min), abs(x_max), 1.0)
            padding = max(min_padding, 0.1 * center_scale)
        else:
            padding = max(min_padding, span * padding_ratio)

        x_limits.append((x_min - padding, x_max + padding))

    return x_limits

def plot_combined_posteriors(all_model_samples, theta_true, output_dir, filename="combined_posteriors.png", sample_spaces=None):
    """
    Plots posteriors from multiple models on one figure.
    all_model_samples: dict {model_name: samples}
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_model_samples:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_samples = next(iter(all_model_samples.values()))
    first_key = next(iter(all_model_samples.keys()))
    first_samples = _convert_samples_for_plot(
        first_samples,
        sample_space=_resolve_sample_space(sample_spaces, first_key),
        target_space="log",
    )
    dims_to_plot = list(range(min(4, first_samples.shape[1])))
    param_names = [rf"$\log(\theta_{{{i+1}}})$" for i in dims_to_plot]

    colors = {
        "SMMD": "#DB1218",
        "MMD": "#1083CA",
        "BAYESFLOW": "#49C926",
        "DNNABC": "#8315dd",
        "NPE": "#f19327",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    sample_groups = [
        _convert_samples_for_plot(
            samples,
            sample_space=_resolve_sample_space(sample_spaces, model_name),
            target_space="log",
        )
        for model_name, samples in all_model_samples.items()
        if samples is not None
    ]
    x_limits = _compute_adaptive_xlim_per_dim(sample_groups, theta_true)

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        for model_name, samples in all_model_samples.items():
            if samples is None:
                continue
            samples = _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="log",
            )

            label_upper = str(model_name).upper()
            c = "#333333"
            if "SMMD" in label_upper:
                c = colors["SMMD"]
            elif "MMD" in label_upper:
                c = colors["MMD"]
            elif "BAYESFLOW" in label_upper:
                c = colors["BAYESFLOW"]
            elif "DNN" in label_upper:
                c = colors["DNNABC"]
            elif "NPE" in label_upper or "SNPE" in label_upper:
                c = colors["NPE"]

            sns.kdeplot(
                x=samples[:, dim_idx],
                label=label_upper,
                ax=ax,
                color=c,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true):
            ax.axvline(
                x=theta_true[dim_idx],
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ["SMMD", "MMD", "BAYESFLOW", "DNNABC", "NPE", "True Value"]
    handles_sorted = []
    labels_sorted = []
    for l in order:
        for k in by_label.keys():
            if l == "True Value" and k == "True Value":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l in k and k not in labels_sorted and "True Value" not in k:
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels
    fig.legend(
        handles_sorted,
        labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(handles_sorted), 5),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )

    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()


def plot_combined_posteriors_theta(all_model_samples, theta_true, output_dir, filename="combined_posteriors_theta.png", sample_spaces=None):
    os.makedirs(output_dir, exist_ok=True)
    if not all_model_samples:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_samples = next(iter(all_model_samples.values()))
    first_key = next(iter(all_model_samples.keys()))
    first_samples = _convert_samples_for_plot(
        first_samples,
        sample_space=_resolve_sample_space(sample_spaces, first_key),
        target_space="theta",
    )
    dims_to_plot = list(range(min(4, first_samples.shape[1])))
    param_names = [rf"$\theta_{{{i+1}}}$" for i in dims_to_plot]

    colors = {
        "SMMD": "#DB1218",
        "MMD": "#1083CA",
        "BAYESFLOW": "#49C926",
        "DNNABC": "#8315dd",
        "NPE": "#f19327",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24

    theta_true = np.asarray(theta_true, dtype=float)
    theta_true_plot = np.exp(np.clip(theta_true, -50.0, 50.0))
    theta_sample_groups = []
    for model_name, samples in all_model_samples.items():
        if samples is None:
            continue
        theta_sample_groups.append(
            _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="theta",
            )
        )
    x_limits = _compute_adaptive_xlim_per_dim(theta_sample_groups, theta_true_plot, min_padding=0.02)

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        for model_name, samples in all_model_samples.items():
            if samples is None:
                continue
            samples_plot = _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="theta",
            )

            label_upper = str(model_name).upper()
            c = "#333333"
            if "SMMD" in label_upper:
                c = colors["SMMD"]
            elif "MMD" in label_upper:
                c = colors["MMD"]
            elif "BAYESFLOW" in label_upper:
                c = colors["BAYESFLOW"]
            elif "DNN" in label_upper:
                c = colors["DNNABC"]
            elif "NPE" in label_upper or "SNPE" in label_upper:
                c = colors["NPE"]

            sns.kdeplot(
                x=samples_plot[:, dim_idx],
                label=label_upper,
                ax=ax,
                color=c,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true_plot):
            ax.axvline(
                x=theta_true_plot[dim_idx],
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = ["SMMD", "MMD", "BAYESFLOW", "DNNABC", "NPE", "True Value"]
    handles_sorted = []
    labels_sorted = []
    for l in order:
        for k in by_label.keys():
            if l == "True Value" and k == "True Value":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l in k and k not in labels_sorted and "True Value" not in k:
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels
    fig.legend(
        handles_sorted,
        labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(handles_sorted), 5),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )

    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

def plot_refinement_comparison(samples_dict, theta_true, output_dir, model_name, sample_spaces=None):
    """
    Plots Initial vs Sequential vs RefinePlus for a single model.
    samples_dict: { 'Initial': samples, 'Sequential': samples, 'RefinePlus': samples }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_refinement_comparison.png"

    valid_samples = [
        _convert_samples_for_plot(
            samples,
            sample_space=_resolve_sample_space(sample_spaces, label),
            target_space="log",
        )
        for label, samples in samples_dict.items()
        if samples is not None
    ]
    if not valid_samples:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    d_local = valid_samples[0].shape[1]
    dims_to_plot = list(range(min(4, d_local)))
    param_names = [rf"$\log(\theta_{{{i+1}}})$" for i in dims_to_plot]

    colors = {
        "SMMD (RefinePlus)": "#FF3A20",
        "SMMD (Baseline)": "#DB1218",
        "MMD (RefinePlus)": "#3FA7D6",
        "MMD (Baseline)": "#1083CA",
        "BAYESFLOW (RefinePlus)": "#49C926",
        "BAYESFLOW (Baseline)": "#2E7D32",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    x_limits = _compute_adaptive_xlim_per_dim(valid_samples, theta_true)

    model_upper = str(model_name).upper()
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        for label, samples in samples_dict.items():
            if samples is None:
                continue
            samples = _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, label),
                target_space="log",
            )

            label_lower = str(label).lower()
            is_amortized = label_lower.startswith("initial") or label_lower.startswith("amort")
            is_sequential = "sequential" in label_lower
            is_refineplus = "refineplus" in label_lower
            if "SMMD" in model_upper:
                c = colors["SMMD (Baseline)" if is_amortized else "SMMD (RefinePlus)"]
            elif "MMD" in model_upper:
                c = colors["MMD (Baseline)" if is_amortized else "MMD (RefinePlus)"]
            else:
                c = colors["BAYESFLOW (Baseline)" if is_amortized else "BAYESFLOW (RefinePlus)"]

            if is_amortized:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_upper} (Amortized)",
                    ax=ax,
                    color=c,
                    fill=False,
                    linestyle="--",
                    linewidth=3.5,
                )
            elif is_refineplus:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_upper} (RefinePlus)",
                    ax=ax,
                    color=c,
                    fill=True,
                    alpha=0.22,
                    linewidth=4.5,
                )
            elif is_sequential:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_upper} (Sequential)",
                    ax=ax,
                    color=c,
                    fill=False,
                    linestyle="-.",
                    linewidth=3.5,
                )
            else:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_upper} ({label})",
                    ax=ax,
                    color=c,
                    fill=True,
                    alpha=0.15,
                    linewidth=4.5,
                )


        if dim_idx < len(theta_true):
            ax.axvline(
                x=theta_true[dim_idx],
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    possible = [
        f"{model_upper} (RefinePlus)",
        f"{model_upper} (Sequential)",
        f"{model_upper} (Amortized)",
        "True Value",
    ]
    sorted_handles = []
    sorted_labels = []
    for k in possible:
        if k in by_label:
            sorted_handles.append(by_label[k])
            sorted_labels.append(k)
    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(sorted_handles), 4),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

def plot_all_refinement_comparisons(all_results_dict, theta_true, output_dir, filename="all_models_refinement_comparison.png"):
    os.makedirs(output_dir, exist_ok=True)
    if not all_results_dict:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_model, first_result = next(iter(all_results_dict.items()))
    first_amortized = _convert_samples_for_plot(
        first_result["Amortized"],
        sample_space=_resolve_sample_space(first_result.get("sample_spaces"), "Amortized"),
        target_space="log",
    )
    d_local = first_amortized.shape[1]
    dims_to_plot = list(range(min(4, d_local)))
    param_names = [rf"$\log(\theta_{{{i+1}}})$" for i in dims_to_plot]

    colors = {
        "SMMD (RefinePlus)": "#FF3A20",
        "SMMD (Baseline)": "#DB1218",
        "MMD (RefinePlus)": "#3FA7D6",
        "MMD (Baseline)": "#1083CA",
        "BAYESFLOW (RefinePlus)": "#49C926",
        "BAYESFLOW (Baseline)": "#2E7D32",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    sample_groups = []
    for result_dict in all_results_dict.values():
        if result_dict is None:
            continue
        sample_groups.extend([
            _convert_samples_for_plot(
                result_dict.get("Amortized"),
                sample_space=_resolve_sample_space(result_dict.get("sample_spaces"), "Amortized"),
                target_space="log",
            ) if result_dict.get("Amortized") is not None else None,
            _convert_samples_for_plot(
                result_dict.get("RefinePlus"),
                sample_space=_resolve_sample_space(result_dict.get("sample_spaces"), "RefinePlus"),
                target_space="log",
            ) if result_dict.get("RefinePlus") is not None else None,
        ])
    x_limits = _compute_adaptive_xlim_per_dim(sample_groups, theta_true)

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        for model_name, result_dict in all_results_dict.items():
            if result_dict is None:
                continue
            amortized = result_dict.get("Amortized")
            refineplus = result_dict.get("RefinePlus")
            if amortized is None or refineplus is None:
                continue

            sample_spaces = result_dict.get("sample_spaces")
            amortized = _convert_samples_for_plot(
                amortized,
                sample_space=_resolve_sample_space(sample_spaces, "Amortized"),
                target_space="log",
            )
            refineplus = _convert_samples_for_plot(
                refineplus,
                sample_space=_resolve_sample_space(sample_spaces, "RefinePlus"),
                target_space="log",
            )

            if model_name.lower() == "smmd":
                c_base = colors["SMMD (Baseline)"]
                c_ref = colors["SMMD (RefinePlus)"]
                label = "SMMD"
            elif model_name.lower() == "mmd":
                c_base = colors["MMD (Baseline)"]
                c_ref = colors["MMD (RefinePlus)"]
                label = "MMD"
            elif model_name.lower() == "bayesflow":
                c_base = colors["BAYESFLOW (Baseline)"]
                c_ref = colors["BAYESFLOW (RefinePlus)"]
                label = "BAYESFLOW"
            else:
                continue

            sns.kdeplot(
                x=amortized[:, dim_idx],
                label=f"{label} (Amortized)",
                ax=ax,
                color=c_base,
                fill=False,
                linestyle="--",
                linewidth=3.5,
            )
            sns.kdeplot(
                x=refineplus[:, dim_idx],
                label=f"{label} (RefinePlus)",
                ax=ax,
                color=c_ref,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true):
            ax.axvline(
                x=theta_true[dim_idx],
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "SMMD (RefinePlus)", "SMMD (Amortized)",
        "MMD (RefinePlus)", "MMD (Amortized)",
        "BAYESFLOW (RefinePlus)", "BAYESFLOW (Amortized)",
        "True Value",
    ]
    sorted_handles = []
    sorted_labels = []
    for l in order:
        if l in by_label:
            sorted_handles.append(by_label[l])
            sorted_labels.append(l)
    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(sorted_handles), 4),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

def plot_all_refinement_comparisons_theta(all_results_dict, theta_true, output_dir, filename="all_models_refinement_comparison_theta.png"):
    os.makedirs(output_dir, exist_ok=True)
    if not all_results_dict:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_model, first_result = next(iter(all_results_dict.items()))
    first_amortized = _convert_samples_for_plot(
        first_result["Amortized"],
        sample_space=_resolve_sample_space(first_result.get("sample_spaces"), "Amortized"),
        target_space="theta",
    )
    d_local = first_amortized.shape[1]
    dims_to_plot = list(range(min(4, d_local)))
    param_names = [rf"$\theta_{{{i+1}}}$" for i in dims_to_plot]

    colors = {
        "SMMD (RefinePlus)": "#FF3A20",
        "SMMD (Baseline)": "#DB1218",
        "MMD (RefinePlus)": "#3FA7D6",
        "MMD (Baseline)": "#1083CA",
        "BAYESFLOW (RefinePlus)": "#49C926",
        "BAYESFLOW (Baseline)": "#2E7D32",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24

    theta_true = np.asarray(theta_true, dtype=float)
    theta_true_plot = np.exp(np.clip(theta_true, -50.0, 50.0))
    theta_sample_groups = []
    for result_dict in all_results_dict.values():
        if result_dict is None:
            continue
        for key in ["Amortized", "RefinePlus"]:
            samples = result_dict.get(key)
            if samples is None:
                continue
            theta_sample_groups.append(
                _convert_samples_for_plot(
                    samples,
                    sample_space=_resolve_sample_space(result_dict.get("sample_spaces"), key),
                    target_space="theta",
                )
            )
    x_limits = _compute_adaptive_xlim_per_dim(theta_sample_groups, theta_true_plot, min_padding=0.02)

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        for model_name, result_dict in all_results_dict.items():
            if result_dict is None:
                continue
            amortized = result_dict.get("Amortized")
            refineplus = result_dict.get("RefinePlus")
            if amortized is None or refineplus is None:
                continue

            sample_spaces = result_dict.get("sample_spaces")
            amortized = _convert_samples_for_plot(
                amortized,
                sample_space=_resolve_sample_space(sample_spaces, "Amortized"),
                target_space="theta",
            )
            refineplus = _convert_samples_for_plot(
                refineplus,
                sample_space=_resolve_sample_space(sample_spaces, "RefinePlus"),
                target_space="theta",
            )

            if model_name.lower() == "smmd":
                c_base = colors["SMMD (Baseline)"]
                c_ref = colors["SMMD (RefinePlus)"]
                label = "SMMD"
            elif model_name.lower() == "mmd":
                c_base = colors["MMD (Baseline)"]
                c_ref = colors["MMD (RefinePlus)"]
                label = "MMD"
            elif model_name.lower() == "bayesflow":
                c_base = colors["BAYESFLOW (Baseline)"]
                c_ref = colors["BAYESFLOW (RefinePlus)"]
                label = "BAYESFLOW"
            else:
                continue

            sns.kdeplot(
                x=amortized[:, dim_idx],
                label=f"{label} (Amortized)",
                ax=ax,
                color=c_base,
                fill=False,
                linestyle="--",
                linewidth=3.5,
            )
            sns.kdeplot(
                x=refineplus[:, dim_idx],
                label=f"{label} (RefinePlus)",
                ax=ax,
                color=c_ref,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true_plot):
            ax.axvline(
                x=theta_true_plot[dim_idx],
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "SMMD (RefinePlus)", "SMMD (Amortized)",
        "MMD (RefinePlus)", "MMD (Amortized)",
        "BAYESFLOW (RefinePlus)", "BAYESFLOW (Amortized)",
        "True Value",
    ]
    sorted_handles = []
    sorted_labels = []
    for l in order:
        if l in by_label:
            sorted_handles.append(by_label[l])
            sorted_labels.append(l)
    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(sorted_handles), 4),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

def plot_final_posteriors_logtheta(final_model_samples, theta_true, output_dir, filename="final_methods_logtheta.png", sample_spaces=None):
    os.makedirs(output_dir, exist_ok=True)
    if not final_model_samples:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_samples = next(iter(final_model_samples.values()))
    first_key = next(iter(final_model_samples.keys()))
    first_samples = _convert_samples_for_plot(
        first_samples,
        sample_space=_resolve_sample_space(sample_spaces, first_key),
        target_space="log",
    )
    dims_to_plot = list(range(min(4, first_samples.shape[1])))
    param_names = [rf"$\log(\theta_{{{i+1}}})$" for i in dims_to_plot]

    colors = {
        "SMMD": "#DB1218",
        "MMD": "#1083CA",
        "BAYESFLOW": "#49C926",
        "DNNABC": "#8315dd",
        "NPE": "#f19327",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    x_limits = _compute_adaptive_xlim_per_dim(
        [
            _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="log",
            )
            for model_name, samples in final_model_samples.items()
            if samples is not None
        ],
        theta_true,
    )

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        for model_name, samples in final_model_samples.items():
            if samples is None:
                continue
            samples = _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="log",
            )

            raw_label = str(model_name)
            label_upper = raw_label.upper()
            is_refineplus = "REFINEPLUS" in label_upper
            c = "#333333"
            if "SMMD" in label_upper:
                base = "SMMD"
                c = colors["SMMD"]
            elif "MMD" in label_upper:
                base = "MMD"
                c = colors["MMD"]
            elif "BAYESFLOW" in label_upper:
                base = "BayesFlow"
                c = colors["BAYESFLOW"]
            elif "DNN" in label_upper:
                base = "DNNABC"
                c = colors["DNNABC"]
            elif "NPE" in label_upper or "SNPE" in label_upper:
                base = "NPE"
                c = colors["NPE"]
            else:
                base = raw_label

            label_display = f"{base} (RefinePlus)" if is_refineplus else base

            sns.kdeplot(
                x=samples[:, dim_idx],
                label=label_display,
                ax=ax,
                color=c,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true):
            ax.axvline(
                x=float(theta_true[dim_idx]),
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "SMMD (RefinePlus)",
        "MMD (RefinePlus)",
        "BayesFlow (RefinePlus)",
        "SMMD",
        "MMD",
        "BayesFlow",
        "DNNABC",
        "NPE",
        "True Value",
    ]
    handles_sorted = []
    labels_sorted = []
    for l in order:
        if l in by_label:
            handles_sorted.append(by_label[l])
            labels_sorted.append(l)
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels
    fig.legend(
        handles_sorted,
        labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(handles_sorted), 5),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )

    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

def plot_final_posteriors_theta(final_model_samples, theta_true, output_dir, filename="final_methods_theta.png", sample_spaces=None):
    os.makedirs(output_dir, exist_ok=True)
    if not final_model_samples:
        return

    sns.set_theme(style="whitegrid", font_scale=1.5)

    first_samples = next(iter(final_model_samples.values()))
    first_key = next(iter(final_model_samples.keys()))
    first_samples = _convert_samples_for_plot(
        first_samples,
        sample_space=_resolve_sample_space(sample_spaces, first_key),
        target_space="theta",
    )
    dims_to_plot = list(range(min(4, first_samples.shape[1])))
    param_names = [rf"$\theta_{{{i+1}}}$" for i in dims_to_plot]

    colors = {
        "SMMD": "#DB1218",
        "MMD": "#1083CA",
        "BAYESFLOW": "#49C926",
        "DNNABC": "#8315dd",
        "NPE": "#f19327",
        "TRUE": "#FA0101",
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24

    theta_true = np.asarray(theta_true, dtype=float)
    theta_true_plot = np.exp(np.clip(theta_true, -50.0, 50.0))
    theta_sample_groups = []
    for model_name, samples in final_model_samples.items():
        if samples is None:
            continue
        theta_sample_groups.append(
            _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="theta",
            )
        )
    x_limits = _compute_adaptive_xlim_per_dim(theta_sample_groups, theta_true_plot, min_padding=0.02)

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        for model_name, samples in final_model_samples.items():
            if samples is None:
                continue
            samples_plot = _convert_samples_for_plot(
                samples,
                sample_space=_resolve_sample_space(sample_spaces, model_name),
                target_space="theta",
            )

            raw_label = str(model_name)
            label_upper = raw_label.upper()
            is_refineplus = "REFINEPLUS" in label_upper
            c = "#333333"
            if "SMMD" in label_upper:
                base = "SMMD"
                c = colors["SMMD"]
            elif "MMD" in label_upper:
                base = "MMD"
                c = colors["MMD"]
            elif "BAYESFLOW" in label_upper:
                base = "BayesFlow"
                c = colors["BAYESFLOW"]
            elif "DNN" in label_upper:
                base = "DNNABC"
                c = colors["DNNABC"]
            elif "NPE" in label_upper or "SNPE" in label_upper:
                base = "NPE"
                c = colors["NPE"]
            else:
                base = raw_label

            label_display = f"{base} (RefinePlus)" if is_refineplus else base

            sns.kdeplot(
                x=samples_plot[:, dim_idx],
                label=label_display,
                ax=ax,
                color=c,
                fill=True,
                alpha=0.15,
                linewidth=4.5,
            )

        if dim_idx < len(theta_true_plot):
            ax.axvline(
                x=float(theta_true_plot[dim_idx]),
                color=colors["TRUE"],
                linestyle="-",
                linewidth=4.0,
                label="True Value",
                alpha=0.95,
            )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, width=2, length=6)
        if i < len(x_limits) and x_limits[i] is not None:
            ax.set_xlim(x_limits[i])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [
        "SMMD (RefinePlus)",
        "MMD (RefinePlus)",
        "BayesFlow (RefinePlus)",
        "SMMD",
        "MMD",
        "BayesFlow",
        "DNNABC",
        "NPE",
        "True Value",
    ]
    handles_sorted = []
    labels_sorted = []
    for l in order:
        if l in by_label:
            handles_sorted.append(by_label[l])
            labels_sorted.append(l)
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels
    fig.legend(
        handles_sorted,
        labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(handles_sorted), 5),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )

    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()

# ============================================================================
# 3. Experiment Runner
# ============================================================================

def run_single_experiment(model_type, task, train_loader, theta_true, x_obs, round_id, model_config={}, initial_training_data=None):
    """
    Runs a single experiment: Train -> Sample -> Compute Metrics
    """
    print(f"\n=== Running {model_type.upper()} (Round {round_id}) ===")
    
    epochs = model_config.get("epochs", 30)
    
    model = None
    loss_history = []
    training_time = 0.0
    
    timings = {"initial": 0.0, "refined_plus": 0.0, "mcmc": 0.0}
    samples_dict = {}
    sample_spaces = {}
    
    n_obs_curr = x_obs.shape[0] # Time steps
    
    # 1. Train
    if model_type == "smmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = SMMD_Model(
            summary_dim=summary_dim,
            d=d,
            d_x=d_x,
            n=n_obs_curr,
            theta_lower=task.lower,
            theta_upper=task.upper,
        )
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "smmd", n_time_steps=n_obs_curr)
    elif model_type == "mmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = MMD_Model(
            summary_dim=summary_dim,
            d=d,
            d_x=d_x,
            n=n_obs_curr,
            theta_lower=task.lower,
            theta_upper=task.upper,
        )
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "mmd", n_time_steps=n_obs_curr)
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            bf_device = torch.device("cpu") # Force CPU for Keras
            summary_dim = model_config.get("summary_dim", 10)
            model, loss_history, training_time = train_bayesflow(
                train_loader,
                epochs,
                bf_device,
                summary_dim=summary_dim,
                theta_lower=task.lower,
                theta_upper=task.upper,
                l1_lambda=float(model_config.get("l1_lambda", 0.0)),
                weight_decay=float(model_config.get("weight_decay", 0.0)),
                learning_rate=model_config.get("learning_rate", LEARNING_RATE),
            )
        else:
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x)
        loss_history = train_dnnabc(model, train_loader, epochs, DEVICE)
        training_time = 0.0  # DNN-ABC training time is not collected here.
    elif model_type in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"]:
        # SBI Models train and sample in one go (for this implementation)
        pass
    
    # 2. Sample (Amortized Inference / ABC)
    print(f"--- Sampling ({model_type}) ---")
    n_samples = N_SAMPLES_POSTERIOR
    
    t_start_sample = time.time()
    
    if model_type in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"]:
        num_rounds = model_config.get("sbi_rounds", 1)
        sims_per_round = model_config.get("sims_per_round", 1000)
        max_epochs = model_config.get("epochs", 1000)
        sample_with = None
        if os.environ.get("LV_QUICK_TEST") == "1":
            sample_with = "mcmc"
        posterior_samples = run_sbi_model(
            model_type, train_loader, x_obs, theta_true, task,
            device=DEVICE, num_rounds=num_rounds,
            sims_per_round=sims_per_round, max_epochs=max_epochs,
            n_posterior_samples=n_samples,
            sample_with=sample_with,
            theta_space="scaled",
            preprocess_obs_fn=preprocess_observations,
            initial_training_data=initial_training_data
        )
    elif model_type == "dnnabc":
        n_pool = model_config.get("n_pool", 100000)
        posterior_samples = abc_rejection_sampling(
            model,
            x_obs,
            task,
            n_samples=n_samples,
            n_pool=n_pool,
            device=DEVICE,
            preprocess_obs_fn=preprocess_observations,
        )
    elif model_type == "bayesflow":
        x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
        if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
        
        # BayesFlow Sample
        post = model.sample(conditions={"summary_variables": x_obs_cpu}, num_samples=n_samples)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if hasattr(post, "numpy"):
            post = post.numpy()
        posterior_samples = np.asarray(post).reshape(-1, d)
        posterior_samples = unconstrained_to_box(posterior_samples, model.theta_lower, model.theta_upper)
        
        std_per_dim = np.std(posterior_samples, axis=0)
        max_abs = np.max(np.abs(posterior_samples), axis=0)
        has_nan = not np.isfinite(posterior_samples).all()
        zero_var = np.any(std_per_dim < 1e-6)
        extreme = np.any(max_abs > 1e3)
        if has_nan or zero_var or extreme:
            print("BayesFlow posterior samples failed sanity check.")
            metrics = {
                "stage": "initial",
                "bias_l2": np.nan,
                "hdi_length": np.full(d, np.nan),
                "coverage": np.full(d, np.nan),
                "training_time": training_time,
            }
            timings["initial"] = training_time
            return {
                "metrics": metrics,
                "metrics_initial": metrics,
                "timings": timings,
                "samples": {"initial": posterior_samples},
                "sample_spaces": {"initial": "log"},
            }
    elif hasattr(model, "sample_posterior"):
        posterior_samples = model.sample_posterior(x_obs, n_samples)
        if isinstance(posterior_samples, torch.Tensor):
            posterior_samples = posterior_samples.detach().cpu().numpy()
    else:
        print("Model cannot sample.")
        return None
        
    sampling_time = time.time() - t_start_sample
    
    # 3. Compute Metrics (Initial posterior)
    metrics = compute_metrics(posterior_samples, theta_true)
    metrics["training_time"] = training_time
    metrics["sampling_time"] = sampling_time
    metrics["stage"] = "initial"
    metrics_initial = metrics.copy()
    
    # Store Initial: total wall-clock for initial posterior (train + sample)
    timings["initial"] = training_time + sampling_time
    samples_dict["initial"] = posterior_samples
    sample_spaces["initial"] = "log"
    
    # Save Samples & Plot
    # Structure: results/models/{model_type}/round_{round_id}/initial/
    base_dir = f"{RESULTS_BASE}/models/{model_type}/round_{round_id}/initial"
    os.makedirs(base_dir, exist_ok=True)
    
    np.save(f"{base_dir}/posterior_samples.npy", posterior_samples)
    plot_posterior(posterior_samples, theta_true, base_dir, "posterior_plot.png", sample_space="log", plot_space="log")
    
    print(f"Bias L2: {metrics['bias_l2']:.4f}")
    print(f"HDI Length per param: {metrics['hdi_length']}")
    print(f"Coverage per param: {metrics['coverage']}")
    
    # ========================================================================
    # 4. Save Initial Model
    # ========================================================================
    save_model_dir = f"{RESULTS_BASE}/models/{model_type}/round_{round_id}"
    if SAVE_MODEL_PARAMS:
        os.makedirs(save_model_dir, exist_ok=True)
        if model_type == "bayesflow":
            try:
                model.save_weights(f"{save_model_dir}/initial_weights.weights.h5")
            except Exception as e:
                print(f"Failed to save BayesFlow weights: {e}")
        elif model_type in ["smmd", "mmd"]:
            torch.save(model.state_dict(), f"{save_model_dir}/initial_state_dict.pt")
        
    # ========================================================================
    # 5. Sequential Training (intermediate stage before RefinePlus MCMC)
    # ========================================================================
    run_refined_training = model_type in ["smmd", "mmd", "bayesflow"]
    
    # Store theta_init for MCMC
    theta_init_mcmc = None
    metrics_sequential = None
    
    if run_refined_training:
        print(f"\n--- Starting Sequential Training for {model_type} ---")
        
        # rounds of sequential retraining
        num_refine_rounds = model_config.get("num_refine_rounds", 1)
        
        for refine_round_idx in range(1, num_refine_rounds + 1):
            print(f"\n[Sequential Round {refine_round_idx}/{num_refine_rounds}]")
            
            # Determine Refined+ Mode
            refined_mode = model_config.get("refined_mode", 1)
            
            if refined_mode == 0:
                print(f"Refined+ Mode is 0. Skipping sequential training.")
                break
                
            N_new = int(model_config.get("refine_train_samples", SMMD_REFINE_TRAIN_SAMPLES))
            kde_n_samples = int(model_config.get("kde_n_samples", 10_000))
            sequential_resample_pool_size = int(model_config.get("sequential_resample_pool_size", 10_000))
            posterior_sample_batch_size = int(model_config.get("posterior_sample_batch_size", 1024))
            sequential_mixture_coeff = float(model_config.get("sequential_mixture_coeff", 0.5))
            print(f"1. Sampling {N_new} parameters from current posterior...")
            
            def sample_posterior_logtheta(n_draw: int) -> np.ndarray:
                parts = []
                remaining = int(n_draw)
                while remaining > 0:
                    bsz = min(posterior_sample_batch_size, remaining)
                    if model_type == "bayesflow":
                        x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.detach().cpu().numpy()
                        if x_obs_cpu.ndim == 2:
                            x_obs_cpu = x_obs_cpu[np.newaxis, ...]
                        x_obs_rep = np.tile(x_obs_cpu, (bsz, 1, 1))
                        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
                        if isinstance(post, dict):
                            post = post["inference_variables"]
                        if hasattr(post, "numpy"):
                            post = post.numpy()
                        theta_b = np.asarray(post).reshape(bsz, -1)
                        theta_b = unconstrained_to_box(theta_b, model.theta_lower, model.theta_upper)
                    elif hasattr(model, "sample_posterior"):
                        theta_b = model.sample_posterior(x_obs, bsz)
                    else:
                        raise RuntimeError("Model cannot sample posterior.")

                    if isinstance(theta_b, torch.Tensor):
                        theta_b = theta_b.detach().cpu().numpy()
                    theta_b = np.asarray(theta_b)
                    if theta_b.ndim == 3 and theta_b.shape[1] == 1:
                        theta_b = theta_b.squeeze(1)

                    parts.append(theta_b)
                    remaining -= bsz

                return np.concatenate(parts, axis=0)

            theta_new = sample_posterior_logtheta(N_new)
                
            if isinstance(theta_new, torch.Tensor):
                theta_new = theta_new.cpu().numpy()
            
            if theta_new is None:
                print("Failed to sample for sequential stage. Skipping.")
                continue
                
            print(f"2. Simulating new data (N={N_new})...")

            # Simulate x from q_train
            x_new = preprocess_observations(task.simulator(theta_new))
             
            # Prior Data (Reuse old training data)
            print(f"3. Reusing {N_new} samples from initial training data...")
            # Reuse original training data (theta_train, x_train)
            # Need to ensure N_new matches or we subsample/repeat
            # train_loader has the data. Let's extract it or use the variables from main loop if passed.
            # But run_single_experiment takes train_loader.
            # We can iterate train_loader to get all data.
            
            theta_old_list = []
            x_old_list = []
            for batch in train_loader:
                # batch is usually (x, theta) or (theta, x) depending on creation
                # TensorDataset(x_train_tensor, theta_train_tensor)
                # So batch[0] is x, batch[1] is theta
                x_b, t_b = batch[0], batch[1]
                theta_old_list.append(t_b)
                x_old_list.append(x_b)
            
            theta_old = torch.cat(theta_old_list).numpy()
            x_old = torch.cat(x_old_list).numpy()
            
            # Squeeze theta_old if necessary
            # TensorDataset might wrap theta as (batch, d) or (batch, 1, d)
            # We want (N, d)
            if theta_old.ndim == 3 and theta_old.shape[1] == 1:
                theta_old = theta_old.squeeze(1)
                
            # Check theta_new shape (should be (N, d))
            if theta_new.ndim == 3 and theta_new.shape[1] == 1:
                theta_new = theta_new.squeeze(1)
            
            # If dataset size differs from N_new, adjust. 
            # Usually N_new = N (dataset size).
            if len(theta_old) != N_new:
                # Subsample or repeat
                indices = np.random.choice(len(theta_old), size=N_new, replace=True)
                theta_old = theta_old[indices]
                x_old = x_old[indices]

            # Fit a KDE on q_train samples (theta_new) for the later importance weights,
            # approximating the proposal density q_train(theta).
            print("   Fitting boundary-corrected KDE on q_train samples...")
            theta_kde = theta_new
            if kde_n_samples > theta_kde.shape[0]:
                extra = sample_posterior_logtheta(kde_n_samples - theta_kde.shape[0])
                theta_kde = np.concatenate([theta_kde, extra], axis=0)
            else:
                theta_kde = theta_kde[:kde_n_samples]

            kde_q_train_model = BoundaryCorrectedKDE(task.lower, task.upper).fit(theta_kde)
            
            # Combine
            theta_combined = np.concatenate([theta_old, theta_new], axis=0)
            x_combined = np.concatenate([x_old, x_new], axis=0)
            
            # No weights for training (Equation 1 in ReviseComment is sum of losses)
            # We just create a dataset of 2N
            print(f"   Combined Dataset: {len(theta_combined)} samples. (Unweighted for training)")
            
            # D. Retrain
            print("4. Retraining on combined dataset...")
            ds_retrain = TensorDataset(
                torch.from_numpy(x_combined).float(), 
                torch.from_numpy(theta_combined).float()
            )
            loader_retrain = DataLoader(ds_retrain, batch_size=BATCH_SIZE, shuffle=True)
            
            if model_type == "bayesflow":
                 # Retrain BayesFlow
                 # Note: train_bayesflow returns (model, history, time)
                 # We ignore the returned model since it updates in place (Keras)
                 model, loss_hist_retrain, time_retrain = train_bayesflow(
                     loader_retrain,
                     epochs=epochs,
                     device="cpu",
                     summary_dim=model_config.get("summary_dim", 10),
                     theta_lower=task.lower,
                     theta_upper=task.upper,
                     model=model,
                     l1_lambda=float(model_config.get("l1_lambda", 0.0)),
                     weight_decay=float(model_config.get("weight_decay", 0.0)),
                     learning_rate=model_config.get("learning_rate", LEARNING_RATE),
                 )
            else:
                 loss_hist_retrain, time_retrain = train_smmd_mmd(model, loader_retrain, epochs=epochs, device=DEVICE, model_type=model_type)
            
            print(f"   Retraining done in {time_retrain:.2f}s")
            
            timings["refined_plus"] += time_retrain
            
            if SAVE_MODEL_PARAMS:
                if model_type == "bayesflow":
                    try:
                        model.save_weights(f"{save_model_dir}/sl_round{refine_round_idx}.weights.h5")
                    except Exception:
                        pass
                else:
                    torch.save(model.state_dict(), f"{save_model_dir}/sl_state_dict_round{refine_round_idx}.pt")

            # Sample Retrained Model (Mixture Model)
            print(f"5. Sampling Retrained Model (Sequential Round {refine_round_idx})...")
            
            pool_size = max(int(n_samples), sequential_resample_pool_size)
            theta_retrained_pool = sample_posterior_logtheta(pool_size)
            
            # Calculate Weights for Resampling (to recover target posterior)
            # w = prior / (0.5 * q_train + 0.5 * prior)
            # Note: Denominator uses q_train (initial), NOT q_retrained!
            print("6. Calculating weights for resampling...")
            
            # Evaluate q_train density on theta_retrained_pool
            # kde_q_train_model was fitted on theta_new (samples from q_train)
            log_q = np.asarray(kde_q_train_model.score_samples(theta_retrained_pool), dtype=np.float64)
            log_p = np.asarray(task.log_prior(theta_retrained_pool), dtype=np.float64)
            log_denom = np.logaddexp(np.log(1.0 - sequential_mixture_coeff) + log_q, np.log(sequential_mixture_coeff) + log_p)
            log_w = log_p - log_denom
            log_w = np.nan_to_num(log_w, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            max_log_w = np.max(log_w)
            if not np.isfinite(max_log_w):
                weights_resample = np.ones(theta_retrained_pool.shape[0], dtype=np.float64)
            else:
                weights_resample = np.exp(log_w - max_log_w)
            
            # Handle NaNs or Infs
            weights_resample = np.nan_to_num(weights_resample, nan=0.0, posinf=0.0, neginf=0.0)
            
            sum_w = np.sum(weights_resample)
            if sum_w < 1e-9:
                print("Warning: Sum of weights is close to 0. Using uniform weights.")
                weights_resample = np.ones_like(weights_resample) / len(weights_resample)
            else:
                weights_resample /= sum_w
            
            # Resample
            indices = np.random.choice(theta_retrained_pool.shape[0], size=n_samples, replace=True, p=weights_resample)
            theta_resampled = theta_retrained_pool[indices]
            
            # Store for MCMC
            theta_init_mcmc = np.exp(np.clip(theta_resampled, -50.0, 50.0))
            samples_dict["sequential"] = theta_resampled
            sample_spaces["sequential"] = "log"

            metrics_sequential = compute_metrics(theta_resampled, theta_true, sample_space="log")
            metrics_sequential["training_time"] = training_time + timings["refined_plus"]
            metrics_sequential["stage"] = "sequential"
            print(f"   Sequential Bias L2: {metrics_sequential['bias_l2']:.4f}")
            print(f"   Sequential HDI Length (Mean): {np.mean(metrics_sequential['hdi_length']):.4f}")
            print(f"   Sequential Coverage (Mean): {np.mean(metrics_sequential['coverage']):.4f}")

            # Sequential posterior is kept in memory for comparison plots and tables,
            # but we do not persist a standalone posterior file/plot for this stage.
            
    # ========================================================================
            
    # ========================================================================
    # 6. ABC-MCMC Refinement (For SMMD/MMD/BayesFlow)
    # ========================================================================
    run_mcmc = (model_type in ["smmd", "mmd", "bayesflow"]) and bool(model_config.get("run_mcmc", True))
    
    if run_mcmc:
        # Check Refined Mode for MCMC
        refined_mode = model_config.get("refined_mode", 1)
        if refined_mode == 0:
            print(f"Refined Mode is 0. Skipping ABC-MCMC for {model_type}.")
            return {
                "metrics": metrics,
                "timings": timings,
                "samples": samples_dict,
                "sample_spaces": sample_spaces,
            }
            
        print(f"\n--- Performing ABC-MCMC Local Refinement for {model_type} ---")
        t_mcmc_start = time.time()
        print(f"   n_sims per proposal: {N_SIMS}")
        
        # Ensure model is in eval mode
        if hasattr(model, "eval"): model.eval()
        
        # Pass n_chains=n_samples and n_samples=1 to get exactly n_samples
        # Pass device explicitly
        mcmc_total_steps = int(model_config.get("mcmc_total_steps", MCMC_TOTAL_STEPS))
        if mcmc_total_steps < 2:
            mcmc_total_steps = 2
        mcmc_burn_in = int(model_config.get("mcmc_burn_in", mcmc_total_steps - 1))
        refine_dir = f"{RESULTS_BASE}/models/{model_type}/round_{round_id}/refineplus"
        os.makedirs(refine_dir, exist_ok=True)
        
        # Pass theta_init_mcmc if available (from Refined+)
        theta_mcmc = refine_posterior(model, x_obs, task=task, n_chains=n_samples, n_samples=1, 
                                     burn_in=mcmc_burn_in, device=DEVICE, theta_init=theta_init_mcmc,
                                     proposal_std=MCMC_PROPOSAL_STD,
                                     bandwidth_n_samples=BANDWIDTH_N_SAMPLES,
                                     bandwidth_quantile_level=BANDWIDTH_QUANTILE_LEVEL,
                                     preprocess_obs_fn=preprocess_observations,
                                     n_sims=N_SIMS,
                                     bandwidth_plot_path=f"{refine_dir}/summary_distance_distribution.png") 
 
        
        time_mcmc = time.time() - t_mcmc_start
        print(f"   ABC-MCMC done in {time_mcmc:.2f}s")
        
        timings["mcmc"] = time_mcmc
        timings["refined_plus"] = timings["refined_plus"] + timings["mcmc"]
        samples_dict["refineplus"] = theta_mcmc
        sample_spaces["refineplus"] = "theta"
        
        # Metrics MCMC
        metrics_refineplus = compute_metrics(theta_mcmc, theta_true, sample_space="theta")
        # Training time includes retrain time if applicable
        total_train_time = training_time
        if run_refined_training:
             # We didn't capture time_retrain variable scope outside if block easily without initialization
             # But we can approximate or use what we have. 
             # Let's just add it if it happened.
             pass 
             
        metrics_refineplus["training_time"] = total_train_time + time_mcmc
        metrics_refineplus["stage"] = "refineplus"
        print(f"   RefinePlus Bias L2: {metrics_refineplus['bias_l2']:.4f}")
        print(f"   RefinePlus HDI Length (Mean): {np.mean(metrics_refineplus['hdi_length']):.4f}")
        print(f"   RefinePlus Coverage (Mean): {np.mean(metrics_refineplus['coverage']):.4f}")
        
        # Save Samples & Plot
        np.save(f"{refine_dir}/posterior_samples.npy", theta_mcmc)
        plot_posterior(theta_mcmc, theta_true, refine_dir, "posterior_plot.png", sample_space="theta", plot_space="theta")
        
        return {
            "metrics": metrics_refineplus,
            "timings": timings,
            "samples": samples_dict,
            "sample_spaces": sample_spaces,
            "metrics_initial": metrics_initial,
            "metrics_sequential": metrics_sequential,
        }

    if metrics_sequential is not None:
        return {
            "metrics": metrics_sequential,
            "timings": timings,
            "samples": samples_dict,
            "sample_spaces": sample_spaces,
            "metrics_initial": metrics_initial,
            "metrics_sequential": metrics_sequential,
        }

    return {
        "metrics": metrics,
        "timings": timings,
        "samples": samples_dict,
        "sample_spaces": sample_spaces,
        "metrics_initial": metrics_initial,
        "metrics_sequential": metrics_sequential,
    }

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/ground_truth", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/calibration", exist_ok=True)
    
    # Use global config
    MODELS = MODELS_TO_RUN
    
    # Aggregated Results
    model_results_table = {m: [] for m in MODELS}
    
    print(f"Starting Multi-Round Experiment (Rounds={NUM_ROUNDS})")
    print(f"Observation preprocessing transform: {OBS_TRANSFORM}")

    def _save_ground_truth(round_idx, theta_true, x_obs):
        np.savez_compressed(
            f"{RESULTS_BASE}/ground_truth/round_{round_idx}.npz",
            theta_true=np.asarray(theta_true, dtype=np.float32),
            x_obs=np.asarray(x_obs, dtype=np.float32),
        )

    def _compute_and_save_calibration():
        summary_rows = []
        for model_name in MODELS:
            stages = ["initial"]
            if model_name in ["smmd", "mmd", "bayesflow"]:
                stages.append("refineplus")

            for stage in stages:
                theta_true_rounds = []
                posterior_rounds = []

                for round_idx in range(1, NUM_ROUNDS + 1):
                    gt_path = f"{RESULTS_BASE}/ground_truth/round_{round_idx}.npz"
                    if not os.path.exists(gt_path):
                        continue
                    gt = np.load(gt_path)
                    theta_true = np.asarray(gt["theta_true"], dtype=np.float32)

                    if stage == "initial":
                        samples_path = (
                            f"{RESULTS_BASE}/models/{model_name}/round_{round_idx}/initial/posterior_samples.npy"
                        )
                    else:
                        samples_path = (
                            f"{RESULTS_BASE}/models/{model_name}/round_{round_idx}/refineplus/posterior_samples.npy"
                        )

                    if not os.path.exists(samples_path):
                        continue

                    samples = np.load(samples_path)
                    if samples is None:
                        continue
                    samples = np.asarray(samples, dtype=np.float32)
                    if samples.ndim != 2:
                        continue
                    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(theta_true)):
                        continue

                    theta_true_rounds.append(theta_true)
                    posterior_rounds.append(samples)

                if len(posterior_rounds) < 2:
                    continue

                posterior_samples_by_round = np.stack(posterior_rounds, axis=0)
                theta_true_by_round = np.stack(theta_true_rounds, axis=0)

                calib = calibration_error_from_rounds(
                    posterior_samples_by_round=posterior_samples_by_round,
                    theta_true_by_round=theta_true_by_round,
                    n_steps=100,
                    posterior_sample_space="theta" if stage == "refineplus" else "log",
                )

                out_npz = f"{RESULTS_BASE}/calibration/{model_name}_{stage}.npz"
                np.savez_compressed(out_npz, **calib)

                row = {
                    "model": model_name,
                    "stage": stage,
                    "n_rounds": calib["n_rounds"],
                    "median_abs_calibration_error_mean": calib["median_abs_calibration_error_mean"],
                }
                per_dim = np.asarray(calib["median_abs_calibration_error_per_dim"], dtype=np.float32)
                for j in range(per_dim.shape[0]):
                    row[f"median_abs_calibration_error_param{j+1}"] = float(per_dim[j])
                summary_rows.append(row)

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                f"{RESULTS_BASE}/calibration/calibration_summary.csv", index=False
            )
    
    for round_idx in range(1, NUM_ROUNDS + 1):
        round_seed = BASE_SEED + round_idx - 1
        np.random.seed(round_seed)
        torch.manual_seed(round_seed)
        torch.cuda.manual_seed_all(round_seed)
        print(f"\n{'='*20} ROUND {round_idx}/{NUM_ROUNDS} | seed={round_seed} {'='*20}")
        
        # 1. Data Generation
        # Calculate t_max based on n_time_steps from config (assuming dt is fixed or from config)
        # n_obs = T/dt + 1 => T = (n_obs - 1) * dt
        dt = DT
        t_max = (N_TIME_STEPS - 1) * dt
        task = LVTask(t_max=t_max, dt=dt)
        
        print(f"Generating training data (N={DATASET_SIZE}, T_MAX={t_max:.1f}, Steps={N_TIME_STEPS})...")
        theta_train = task.sample_prior(DATASET_SIZE, "vague")
        x_train = preprocess_observations(task.simulator(theta_train))
        theta_train_scaled = scale_theta_np(theta_train, task.lower, task.upper).astype(np.float32)
        
        x_tensor = torch.from_numpy(x_train).float()
        theta_tensor = torch.from_numpy(theta_train).float()
        theta_tensor_scaled = torch.from_numpy(theta_train_scaled).float()
        
        smmd_n = min(SMMD_INITIAL_TRAIN_SAMPLES, DATASET_SIZE)
        bayesflow_conf = MODELS_CONFIG.get("bayesflow", {})
        bayesflow_n = min(
            int(bayesflow_conf.get("initial_train_samples", SMMD_INITIAL_TRAIN_SAMPLES)),
            DATASET_SIZE,
        )
        dnnabc_conf = MODELS_CONFIG.get("dnnabc", {})
        dnnabc_train_n = min(dnnabc_conf.get("train_samples", DATASET_SIZE), DATASET_SIZE)
        
        dataset_smmd = TensorDataset(x_tensor[:smmd_n], theta_tensor[:smmd_n])
        train_loader_smmd = DataLoader(dataset_smmd, batch_size=BATCH_SIZE, shuffle=True)

        dataset_bayesflow = TensorDataset(x_tensor[:bayesflow_n], theta_tensor[:bayesflow_n])
        train_loader_bayesflow = DataLoader(dataset_bayesflow, batch_size=BATCH_SIZE, shuffle=True)
        
        dataset_dnnabc = TensorDataset(x_tensor[:dnnabc_train_n], theta_tensor_scaled[:dnnabc_train_n])
        train_loader_dnnabc = DataLoader(dataset_dnnabc, batch_size=BATCH_SIZE, shuffle=True)
        
        dataset_full = TensorDataset(x_tensor, theta_tensor)
        train_loader_full = DataLoader(dataset_full, batch_size=BATCH_SIZE, shuffle=True)

        dataset_full_scaled = TensorDataset(x_tensor, theta_tensor_scaled)
        train_loader_full_scaled = DataLoader(dataset_full_scaled, batch_size=BATCH_SIZE, shuffle=True)
        
        # Use the task's fixed ground-truth parameters so that per-round
        # metrics (interval length, coverage) are comparable across rounds.
        theta_true, x_obs_raw = task.get_ground_truth()
        theta_true = np.asarray(theta_true, dtype=np.float32)
        x_obs = preprocess_observations(x_obs_raw)
        _save_ground_truth(round_idx, theta_true, x_obs)
        
        round_initial_samples = {}
        round_initial_sample_spaces = {}
        round_refinement_data = {}
        round_final_samples = {}
        round_final_sample_spaces = {}
        
        # Run Models
        for model_name in MODELS:
            try:
                model_conf = MODELS_CONFIG.get(model_name, {})
                if model_name in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"] and not model_conf:
                    model_conf = dict(MODELS_CONFIG.get("npe", {}))
                if isinstance(model_conf, int):
                    model_conf = {"epochs": model_conf}
                
                if model_name in ["smmd", "mmd"]:
                    train_loader_model = train_loader_smmd
                elif model_name == "bayesflow":
                    train_loader_model = train_loader_bayesflow
                elif model_name == "dnnabc":
                    train_loader_model = train_loader_dnnabc
                elif model_name in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"]:
                    train_loader_model = train_loader_full_scaled
                else:
                    train_loader_model = train_loader_full
                
                res = run_single_experiment(
                    model_name, task, train_loader_model, theta_true, x_obs, round_idx,
                    model_config=model_conf,
                    initial_training_data=(x_train, theta_train)
                )
                
                if res:
                    metrics = res["metrics"]
                    timings = res["timings"]
                    samples = res["samples"]
                    result_sample_spaces = res.get("sample_spaces", {})
                    
                    if "initial" in samples:
                        init_s = samples["initial"]
                        if isinstance(init_s, torch.Tensor):
                            init_s = init_s.detach().cpu().numpy()
                        round_initial_samples[model_name] = init_s
                        round_initial_sample_spaces[model_name] = result_sample_spaces.get("initial", "log")
                    
                    final_refineplus = None
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        sequential_samples = samples.get("sequential")
                        if isinstance(sequential_samples, torch.Tensor):
                            sequential_samples = sequential_samples.detach().cpu().numpy()

                        final_refineplus = samples.get("refineplus")
                        if isinstance(final_refineplus, torch.Tensor):
                            final_refineplus = final_refineplus.detach().cpu().numpy()

                        if final_refineplus is not None:
                            round_refinement_data[model_name] = {
                                "Amortized": round_initial_samples.get(model_name),
                                "RefinePlus": final_refineplus,
                                "sample_spaces": {
                                    "Amortized": round_initial_sample_spaces.get(model_name, "log"),
                                    "RefinePlus": result_sample_spaces.get("refineplus", "theta"),
                                },
                            }

                            plot_samples = {
                                "Initial": samples.get("initial"),
                                "Sequential": sequential_samples,
                                "RefinePlus": final_refineplus,
                            }
                            plot_samples = {k: v for k, v in plot_samples.items() if v is not None}
                            plot_sample_spaces = {
                                "Initial": result_sample_spaces.get("initial", "log"),
                                "Sequential": result_sample_spaces.get("sequential", "log"),
                                "RefinePlus": result_sample_spaces.get("refineplus", "theta"),
                            }
                            plot_sample_spaces = {
                                k: v for k, v in plot_sample_spaces.items() if k in plot_samples
                            }
                            
                            if SAVE_REFINEMENT_COMPARISON_PLOTS:
                                # Refine+ figure: posterior densities across the
                                # amortized / sequential / refinement stages.
                                plot_dir = f"{RESULTS_BASE}/models/{model_name}/round_{round_idx}/comparison"
                                plot_refinement_comparison(
                                    plot_samples,
                                    theta_true,
                                    plot_dir,
                                    model_name,
                                    sample_spaces=plot_sample_spaces,
                                )

                    if model_name == "smmd":
                        if final_refineplus is not None:
                            round_final_samples["SMMD (RefinePlus)"] = final_refineplus
                            round_final_sample_spaces["SMMD (RefinePlus)"] = result_sample_spaces.get("refineplus", "theta")
                        else:
                            round_final_samples["SMMD"] = round_initial_samples.get(model_name)
                            round_final_sample_spaces["SMMD"] = round_initial_sample_spaces.get(model_name, "log")
                    elif model_name == "mmd":
                        if final_refineplus is not None:
                            round_final_samples["MMD (RefinePlus)"] = final_refineplus
                            round_final_sample_spaces["MMD (RefinePlus)"] = result_sample_spaces.get("refineplus", "theta")
                        else:
                            round_final_samples["MMD"] = round_initial_samples.get(model_name)
                            round_final_sample_spaces["MMD"] = round_initial_sample_spaces.get(model_name, "log")
                    elif model_name == "bayesflow":
                        if final_refineplus is not None:
                            round_final_samples["BAYESFLOW (RefinePlus)"] = final_refineplus
                            round_final_sample_spaces["BAYESFLOW (RefinePlus)"] = result_sample_spaces.get("refineplus", "theta")
                        else:
                            round_final_samples["BAYESFLOW"] = round_initial_samples.get(model_name)
                            round_final_sample_spaces["BAYESFLOW"] = round_initial_sample_spaces.get(model_name, "log")
                    elif model_name == "dnnabc":
                        round_final_samples["DNNABC"] = round_initial_samples.get(model_name)
                        round_final_sample_spaces["DNNABC"] = round_initial_sample_spaces.get(model_name, "log")
                    elif model_name in ["npe", "snpe", "snpe_a", "snpe_b", "snpe_c"]:
                        round_final_samples["NPE"] = round_initial_samples.get(model_name)
                        round_final_sample_spaces["NPE"] = round_initial_sample_spaces.get(model_name, "log")

                    metrics_initial = res.get("metrics_initial")
                    metrics_sequential = res.get("metrics_sequential")
                    
                    row = metrics.copy()
                    row["round"] = round_idx
                    row["seed"] = round_seed
                    row["status"] = 1
                    
                    if metrics_initial is not None:
                        if "bias_l2" in metrics_initial:
                            row["bias_l2_initial"] = metrics_initial["bias_l2"]
                        if "hdi_length" in metrics_initial:
                            row["hdi_length_initial"] = metrics_initial["hdi_length"]
                        if "coverage" in metrics_initial:
                            row["coverage_initial"] = metrics_initial["coverage"]
                    else:
                        if "bias_l2" in metrics:
                            row["bias_l2_initial"] = metrics["bias_l2"]
                        if "hdi_length" in metrics:
                            row["hdi_length_initial"] = metrics["hdi_length"]
                        if "coverage" in metrics:
                            row["coverage_initial"] = metrics["coverage"]
                    
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        if metrics_sequential is not None:
                            if "bias_l2" in metrics_sequential:
                                row["bias_l2_sequential"] = metrics_sequential["bias_l2"]
                            if "hdi_length" in metrics_sequential:
                                row["hdi_length_sequential"] = metrics_sequential["hdi_length"]
                            if "coverage" in metrics_sequential:
                                row["coverage_sequential"] = metrics_sequential["coverage"]
                        if metrics.get("stage") == "refineplus":
                            if "bias_l2" in metrics:
                                row["bias_l2_refineplus"] = metrics["bias_l2"]
                            if "hdi_length" in metrics:
                                row["hdi_length_refineplus"] = metrics["hdi_length"]
                            if "coverage" in metrics:
                                row["coverage_refineplus"] = metrics["coverage"]
                    
                    # Add formatted times
                    row["time_initial"] = format_time(timings["initial"])
                    row["time_refined_plus"] = format_time(timings["refined_plus"])
                    row["time_mcmc"] = format_time(timings["mcmc"])
                    
                    # Keep raw seconds for averaging
                    row["seconds_initial"] = timings["initial"]
                    row["seconds_refined_plus"] = timings["refined_plus"]
                    row["seconds_mcmc"] = timings["mcmc"]
                    
                    model_results_table[model_name].append(row)
                    
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Record failure
                row = {
                    "round": round_idx,
                    "seed": round_seed,
                    "status": 0, # Failure
                    "error_msg": str(e)
                }
                model_results_table[model_name].append(row)

        if round_initial_samples:
            print("Generating combined posterior plot...")
            plot_combined_posteriors(
                round_initial_samples,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                "all_methods_initial_posterior.png",
                sample_spaces=round_initial_sample_spaces,
            )
            plot_combined_posteriors_theta(
                round_initial_samples,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                "all_methods_initial_posterior_theta.png",
                sample_spaces=round_initial_sample_spaces,
            )
        
        if round_refinement_data and SAVE_REFINEMENT_COMPARISON_PLOTS:
            plot_all_refinement_comparisons(
                round_refinement_data,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                filename="all_models_refinement_comparison.png",
            )
            plot_all_refinement_comparisons_theta(
                round_refinement_data,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                filename="all_models_refinement_comparison_theta.png",
            )
            plot_final_posteriors_logtheta(
                round_final_samples,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                filename="all_methods_final_logtheta.png",
                sample_spaces=round_final_sample_spaces,
            )
            plot_final_posteriors_theta(
                round_final_samples,
                theta_true,
                f"{RESULTS_BASE}/comparisons/round_{round_idx}",
                filename="all_methods_final_theta.png",
                sample_spaces=round_final_sample_spaces,
            )

        aggregate_and_save_results(model_results_table)

    # ============================================================================
    # 4. Final Aggregation
    # ============================================================================
    print("\n" + "="*30)
    print("FINAL AGGREGATED RESULTS")
    print("="*30)
    
    aggregate_and_save_results(model_results_table)
    _compute_and_save_calibration()

def aggregate_and_save_results(model_results_table):
    os.makedirs(f"{RESULTS_BASE}/tables", exist_ok=True)
    summary_data = []
    
    for model_name, rows in model_results_table.items():
        if not rows:
            continue
            
        df_model = pd.DataFrame(rows)
        # Table 3: per-replication and aggregated metrics.
        cols = ["round", "status", "stage", "bias_l2", "hdi_length", "coverage", 
                "time_initial", "time_refined_plus", "time_mcmc"]
        other_cols = [c for c in df_model.columns if c not in cols and not c.startswith("seconds_")]
        final_cols = cols + other_cols
        final_cols = [c for c in final_cols if c in df_model.columns]
        df_model = df_model[final_cols]
        df_model.to_csv(f"{RESULTS_BASE}/tables/{model_name}_results.csv", index=False)
        print(f"Saved table for {model_name} to {RESULTS_BASE}/tables/{model_name}_results.csv")
            
        success_rows = [r for r in rows if r.get("status", 0) == 1]
        if not success_rows:
            print(f"No successful runs for {model_name}")
            continue
        
        example = None
        for r in success_rows:
            if "hdi_length_initial" in r or "hdi_length" in r:
                example = r
                break
        if example is None:
            print(f"No HDI information for {model_name}, skipping aggregation.")
            continue
        if "hdi_length_initial" in example:
            example_hdi = np.asarray(example["hdi_length_initial"], dtype=float)
        else:
            example_hdi = np.asarray(example["hdi_length"], dtype=float)
        num_params = example_hdi.shape[0]
        
        per_round_rows = []
        refine_models = ["smmd", "mmd", "bayesflow"]
        
        for r in rows:
            round_id = r.get("round", np.nan)
            status = r.get("status", 0)
            
            if status != 1:
                row_metrics = {
                    "round": round_id,
                    "Bias": np.nan,
                    "RefinePlus_Bias": np.nan,
                }
                for j in range(num_params):
                    row_metrics[f"HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"Coverage_Param{j+1}"] = np.nan
                    row_metrics[f"RefinePlus_HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"RefinePlus_Coverage_Param{j+1}"] = np.nan
                per_round_rows.append(row_metrics)
                continue
            
            if "bias_l2_initial" in r:
                bias_init = r["bias_l2_initial"]
            elif "bias_l2" in r:
                bias_init = r["bias_l2"]
            else:
                bias_init = np.nan
            
            if model_name in refine_models and "bias_l2_refineplus" in r:
                bias_ref = r["bias_l2_refineplus"]
            else:
                bias_ref = np.nan
            
            if "hdi_length_initial" in r:
                hdi_init = np.asarray(r["hdi_length_initial"], dtype=float)
            elif "hdi_length" in r:
                hdi_init = np.asarray(r["hdi_length"], dtype=float)
            else:
                hdi_init = np.full(num_params, np.nan)
            
            if model_name in refine_models and "hdi_length_refineplus" in r:
                hdi_ref = np.asarray(r["hdi_length_refineplus"], dtype=float)
            else:
                hdi_ref = np.full(num_params, np.nan)
            
            if "coverage_initial" in r:
                cov_init = np.asarray(r["coverage_initial"], dtype=float)
            elif "coverage" in r:
                cov_init = np.asarray(r["coverage"], dtype=float)
            else:
                cov_init = np.full(num_params, np.nan)
            
            if model_name in refine_models and "coverage_refineplus" in r:
                cov_ref = np.asarray(r["coverage_refineplus"], dtype=float)
            else:
                cov_ref = np.full(num_params, np.nan)
            
            row_metrics = {
                "round": round_id,
                "Bias": bias_init,
                "RefinePlus_Bias": bias_ref,
            }
            for j in range(num_params):
                row_metrics[f"HDI_Len_Param{j+1}"] = hdi_init[j]
                row_metrics[f"Coverage_Param{j+1}"] = cov_init[j]
                row_metrics[f"RefinePlus_HDI_Len_Param{j+1}"] = hdi_ref[j]
                row_metrics[f"RefinePlus_Coverage_Param{j+1}"] = cov_ref[j]
            
            per_round_rows.append(row_metrics)
        
        per_round_df = pd.DataFrame(per_round_rows)
        model_dir = f"{RESULTS_BASE}/models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        per_round_path = f"{model_dir}/{model_name}_per_round_metrics.csv"
        per_round_df.to_csv(per_round_path, index=False)
        print(f"Saved per-round metrics for {model_name} to {per_round_path}")
        
        bias_initial = []
        bias_refineplus = []
        hdi_initial_list = []
        hdi_refineplus_list = []
        coverage_initial_list = []
        coverage_refineplus_list = []
        
        for r in success_rows:
            if "bias_l2_initial" in r:
                bias_initial.append(r["bias_l2_initial"])
            elif "bias_l2" in r:
                bias_initial.append(r["bias_l2"])
            if model_name in refine_models and "bias_l2_refineplus" in r:
                bias_refineplus.append(r["bias_l2_refineplus"])
            
            if "hdi_length_initial" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length_initial"], dtype=float))
            elif "hdi_length" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length"], dtype=float))
            if model_name in refine_models and "hdi_length_refineplus" in r:
                hdi_refineplus_list.append(np.asarray(r["hdi_length_refineplus"], dtype=float))
            
            if "coverage_initial" in r:
                coverage_initial_list.append(np.asarray(r["coverage_initial"], dtype=float))
            elif "coverage" in r:
                coverage_initial_list.append(np.asarray(r["coverage"], dtype=float))
            if model_name in refine_models and "coverage_refineplus" in r:
                coverage_refineplus_list.append(np.asarray(r["coverage_refineplus"], dtype=float))
        
        hdi_initial_arr = np.vstack(hdi_initial_list)
        avg_hdi_len_initial = hdi_initial_arr.mean(axis=0)
        
        coverage_initial_arr = np.vstack(coverage_initial_list)
        avg_coverage_initial = coverage_initial_arr.mean(axis=0)
        
        if hdi_refineplus_list:
            hdi_refineplus_arr = np.vstack(hdi_refineplus_list)
            avg_hdi_len_refineplus = hdi_refineplus_arr.mean(axis=0)
        else:
            avg_hdi_len_refineplus = np.full_like(avg_hdi_len_initial, np.nan)
        
        if coverage_refineplus_list:
            coverage_refineplus_arr = np.vstack(coverage_refineplus_list)
            avg_coverage_refineplus = coverage_refineplus_arr.mean(axis=0)
        else:
            avg_coverage_refineplus = np.full_like(avg_coverage_initial, np.nan)
        
        avg_sec_initial = np.mean([r["seconds_initial"] for r in success_rows])
        avg_sec_sl = np.mean([r["seconds_refined_plus"] for r in success_rows])
        avg_sec_mcmc = np.mean([r["seconds_mcmc"] for r in success_rows])
        
        mean_bias_initial = np.mean(bias_initial)
        median_bias_initial = np.median(bias_initial)
        
        if bias_refineplus:
            mean_bias_refineplus = np.mean(bias_refineplus)
            median_bias_refineplus = np.median(bias_refineplus)
        else:
            mean_bias_refineplus = np.nan
            median_bias_refineplus = np.nan
        
        print(f"\nModel: {model_name.upper()}")
        print(f"  Bias L2 (Initial): Mean={mean_bias_initial:.4f}")
        if not np.isnan(mean_bias_refineplus):
            print(f"  Bias L2 (Refine+): Mean={mean_bias_refineplus:.4f}")
        print(f"  HDI Len (Initial) per param: {avg_hdi_len_initial}")
        if not np.all(np.isnan(avg_hdi_len_refineplus)):
            print(f"  HDI Len (Refine+) per param: {avg_hdi_len_refineplus}")
        print(f"  Coverage (Initial) per param: {avg_coverage_initial}")
        if not np.all(np.isnan(avg_coverage_refineplus)):
            print(f"  Coverage (Refine+) per param: {avg_coverage_refineplus}")
        print(f"  Avg Time (Initial): {format_time(avg_sec_initial)}")
        if avg_sec_sl > 0:
            print(f"  Avg Time (Refined+): {format_time(avg_sec_sl)}")
        if avg_sec_mcmc > 0:
            print(f"  Avg Time (MCMC): {format_time(avg_sec_mcmc)}")
        
        summary_row = {
            "Model": model_name,
            "Bias_Mean": mean_bias_initial,
            "RefinePlus_Bias_Mean": mean_bias_refineplus,
            "Bias_Median": median_bias_initial,
            "RefinePlus_Bias_Median": median_bias_refineplus,
            "Avg_Time_Initial": format_time(avg_sec_initial),
            "Avg_Time_RefinedPlus": format_time(avg_sec_sl),
            "Avg_Time_MCMC": format_time(avg_sec_mcmc),
        }
        
        for j in range(num_params):
            summary_row[f"HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_initial[j]
            summary_row[f"Coverage_Param{j+1}_Mean"] = avg_coverage_initial[j]
            summary_row[f"RefinePlus_HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_refineplus[j]
            summary_row[f"RefinePlus_Coverage_Param{j+1}_Mean"] = avg_coverage_refineplus[j]
        
        summary_data.append(summary_row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{RESULTS_BASE}/final_summary.csv", index=False)
    print(f"\nSummary saved to {RESULTS_BASE}/final_summary.csv")

if __name__ == "__main__":
    main()
