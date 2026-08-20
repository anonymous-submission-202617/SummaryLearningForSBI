###############################################################################
# Gaussian toy example with multimodal posterior
#
# Trains the proposed summary-learning estimators (SMMD, MMD) and the
# baselines (BayesFlow, DNN-ABC, SNPE-C) at n = 25, 50, 100, runs local
# refinement, and evaluates NLE / TD-NPSE from the released checkpoints.
#
# Reproduces Table 1 ("Results for toy examples."): MMD and sqrt(n) * sW
# between each estimated posterior and the ground-truth posterior, averaged
# over 10 replications.
#
# Outputs: results_<n>/ (see README.md).
###############################################################################

import os
import sys
import atexit
import time
import json
from contextlib import contextmanager
from datetime import datetime
import numpy as np

# Make the experiment package and the repository root importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
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
from sbi_common.config_utils import cli_config_arg, resolve_config_path
from data_generation import GaussianTask, generate_dataset, TRUE_PARAMS, d, d_x, p
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.bayesflow_net import build_bayesflow_model
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.nle import NLESBIOfficialFactorized
from models.sbi_wrappers import run_sbi_model
from models.td_npse import TDNPSE_Model, FullSingleObservationDataset, train_td_npse
from utilities import (
    refine_posterior,
    compute_bandwidth_torch,
    compute_mmd_metric,
    compute_swd_metric,
    run_gaussian_posterior_mcmc,
)


_ORIG_STDOUT = None
_ORIG_STDERR = None
_TERMINAL_LOG_FH = None


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _close_terminal_log():
    global _TERMINAL_LOG_FH
    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
    if _ORIG_STDERR is not None:
        sys.stderr = _ORIG_STDERR
    if _TERMINAL_LOG_FH is None:
        return
    try:
        _TERMINAL_LOG_FH.flush()
        _TERMINAL_LOG_FH.close()
    finally:
        _TERMINAL_LOG_FH = None


def _setup_terminal_log(results_base: str):
    global _ORIG_STDOUT, _ORIG_STDERR, _TERMINAL_LOG_FH
    if _ORIG_STDOUT is None:
        _ORIG_STDOUT = sys.stdout
    if _ORIG_STDERR is None:
        _ORIG_STDERR = sys.stderr

    _close_terminal_log()
    os.makedirs(results_base, exist_ok=True)
    log_path = os.path.join(results_base, "terminal.log")
    _TERMINAL_LOG_FH = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(_ORIG_STDOUT, _TERMINAL_LOG_FH)
    sys.stderr = _Tee(_ORIG_STDERR, _TERMINAL_LOG_FH)
    print(f"[terminal log] {log_path}")


atexit.register(_close_terminal_log)


@contextmanager
def _global_autograd(enabled: bool):
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(enabled)
    try:
        yield
    finally:
        torch.set_grad_enabled(previous)

# Try importing BayesFlow (optional)
try:
    import bayesflow as bf

    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False
    print(
        "BayesFlow not installed or import failed. BayesFlow model will be unavailable."
    )

# ============================================================================
# 1. Hyperparameters & Device
# ============================================================================
# Default values (will be overridden by config.json if available)
M = 50
L = 20
BATCH_SIZE = 256
DATASET_SIZE = 25600
LEARNING_RATE = 1e-3
L1_LAMBDA = 1e-4
NUM_ROUNDS = 5
n = 50
RESULTS_BASE = os.path.join(_SCRIPT_DIR, "results")
N_OBS_LIST = None
N_SAMPLES_POSTERIOR = 2000
RESULTS_SUFFIX = os.environ.get("GAUSSIAN_RESULTS_SUFFIX", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _auto_select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"✅ Auto-detected CUDA acceleration. Device: {device} "
            f"({torch.cuda.get_device_name(0)})"
        )
        return device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Auto-detected MPS (Apple Silicon) acceleration. Device: {device}")
        return device
    device = torch.device("cpu")
    print(f"ℹ️ Auto-detected CPU fallback. Device: {device}")
    return device


def _resolve_device():
    requested = os.environ.get("GAUSSIAN_DEVICE", "").strip().lower()
    if not requested or requested == "auto":
        return _auto_select_device()

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested)
            device_index = device.index if device.index is not None else 0
            print(
                f"✅ Using requested CUDA device: {device} "
                f"({torch.cuda.get_device_name(device_index)})"
            )
            return device
        print(
            f"Warning: requested device '{requested}' but CUDA is not available. "
            "Falling back to automatic device detection."
        )
        return _auto_select_device()

    if requested == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✅ Using requested MPS device: {device}")
            return device
        print(
            "Warning: requested device 'mps' but MPS is not available. "
            "Falling back to automatic device detection."
        )
        return _auto_select_device()

    if requested == "cpu":
        device = torch.device("cpu")
        print(f"ℹ️ Using requested CPU device: {device}")
        return device

    print(
        f"Warning: unsupported GAUSSIAN_DEVICE='{requested}'. "
        "Expected one of: auto, cpu, mps, cuda, cuda:0, cuda:1, ... "
        "Falling back to automatic device detection."
    )
    return _auto_select_device()


# Load Configuration from JSON.
# Selection: --config <path> > GAUSSIAN_CONFIG_PATH env var > config.json.
CONFIG_PATH, _REQUESTED_CONFIG_PATH = resolve_config_path(
    SCRIPT_DIR, "GAUSSIAN_CONFIG_PATH", cli_value=cli_config_arg()
)
if CONFIG_PATH is not None:
    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
else:
    if _REQUESTED_CONFIG_PATH:
        print(
            f"Warning: requested config '{_REQUESTED_CONFIG_PATH}' was not found "
            "in the current machine. Using default parameters."
        )
    else:
        print("Warning: config.json not found. Using default parameters.")
    CONFIG = {}

SAVE_MODEL_PARAMS = bool(CONFIG.get("save_model_params", False))

# Generic experiment-level settings
n = CONFIG.get("n_observation", n)
N_OBS_LIST = CONFIG.get("n_observation_list", None)
NUM_ROUNDS = CONFIG.get("num_rounds", NUM_ROUNDS)
DATASET_SIZE = CONFIG.get("dataset_size", DATASET_SIZE)
BATCH_SIZE = CONFIG.get("batch_size", BATCH_SIZE)
LEARNING_RATE = CONFIG.get("learning_rate", LEARNING_RATE)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", N_SAMPLES_POSTERIOR)

# True posterior sampling config
TRUE_POSTERIOR_CONFIG = CONFIG.get(
    "true_posterior_config",
    {
        "method": "mcmc",
        "n_draws": 1,
        "n_tune_chain": 1000,
        "chains": 1000,
        "proposal_scale_coeff": 0.45,
    },
)

# ABC-MCMC refinement config (shared by SMMD/MMD/BayesFlow)
REFINE_CONFIG = CONFIG.get(
    "refine_config",
    {
        "n_chains": 2000,
        "burn_in": 99,
        "thin": 1,
        "nsims": 1,
        "epsilon": None,
    },
)

RAW_MODELS_CONFIG = CONFIG.get(
    "models_config",
    {
        "smmd": 200,
        "mmd": 200,
        "bayesflow": 200,
        "dnnabc": 200,
        "npe": 1000,
    },
)

MODELS_CONFIG = {}
for name, cfg in RAW_MODELS_CONFIG.items():
    if isinstance(cfg, dict):
        MODELS_CONFIG[name] = cfg.copy()
    else:
        MODELS_CONFIG[name] = {"epochs": cfg}

# Fill per-model defaults
for name in ["smmd", "mmd", "bayesflow"]:
    mc = MODELS_CONFIG.setdefault(name, {})
    mc.setdefault("epochs", 200)
    mc.setdefault("summary_dim", 10)
    mc.setdefault("learning_rate", LEARNING_RATE)
    mc.setdefault("weight_decay", 1e-4)

mc = MODELS_CONFIG.setdefault("dnnabc", {})
mc.setdefault("epochs", 200)

mc = MODELS_CONFIG.setdefault("npe", {})
mc.setdefault("epochs", 1000)
mc.setdefault("sbi_rounds", 5)
mc.setdefault("sims_per_round", 1000)

mc = MODELS_CONFIG.setdefault("nle", {})
mc.setdefault("epochs", 200)
mc.setdefault("learning_rate", 1e-4)
mc.setdefault("hidden_features", 50)
mc.setdefault("num_transforms", 5)
mc.setdefault("num_blocks", 2)
mc.setdefault("z_score_theta", "independent")
mc.setdefault("z_score_x", "independent")
mc.setdefault("validation_split", 0.05)
mc.setdefault("stop_after_epochs", 20)
mc.setdefault("training_batch_size", BATCH_SIZE)
mc.setdefault("observations_per_theta", 10)
mc.setdefault("device", "auto")
mc.setdefault("mcmc_method", "slice_np_vectorized")
mc.setdefault("mcmc_chains", 64)
mc.setdefault("mcmc_chain_batch_size", None)
mc.setdefault("mcmc_warmup_steps", 200)
mc.setdefault("mcmc_thin", 1)
mc.setdefault("slice_width_mode", "constant")
mc.setdefault("slice_init_width", 0.01)
mc.setdefault("slice_reference_n", 50.0)
mc.setdefault("seed", 0)
mc.setdefault("eval_mode", "reload_checkpoint")
mc.setdefault("checkpoint_path", os.path.join("data", "nle_checkpoint.pt"))

mc = MODELS_CONFIG.setdefault("td_npse", {})
mc.setdefault("eval_mode", "reload_checkpoint")
mc.setdefault("checkpoint_path", os.path.join("data", "td_npse_checkpoint.pt"))
mc.setdefault("epochs", 600)
mc.setdefault("learning_rate", 1e-4)
mc.setdefault("hidden_dim", 256)
mc.setdefault("num_layers", 3)
mc.setdefault("hidden_features", [256, 256, 256])
mc.setdefault("time_frequencies", 3)
mc.setdefault("num_diffusion_steps", 100)
mc.setdefault("sample_steps", 1000)
mc.setdefault("sampler_type", "ddim")
mc.setdefault("cov_steps", 100)
mc.setdefault("cov_samples", 1000)
mc.setdefault("ddim_eta", 1.0)
mc.setdefault("cov_reg", 1e-3)
mc.setdefault("cov_mode", "GAUSS")
mc.setdefault("clip_denoised", True)
mc.setdefault("uniform_prior_scale", 2.0)
mc.setdefault("validation_split", 0.2)
mc.setdefault("early_stopping", True)
mc.setdefault("patience", 200)
mc.setdefault("min_nb_epochs", 100)

MODELS_TO_RUN = CONFIG.get("models_to_run", list(MODELS_CONFIG.keys()))
print(f"Configuration loaded: n={n}, rounds={NUM_ROUNDS}, models={MODELS_TO_RUN}")

if N_OBS_LIST is None:
    N_OBS_LIST = [n]

DEVICE = _resolve_device()

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


def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)


def resolve_model_config(model_type):
    model_cfg = MODELS_CONFIG.get(model_type)
    if model_cfg is not None:
        return model_cfg
    if model_type in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"]:
        alias_cfg = MODELS_CONFIG.get("npe")
        if alias_cfg is not None:
            return alias_cfg
    return None


def _resolve_checkpoint_path(cfg):
    """Resolve a model checkpoint path configured in ``cfg`` to an existing file.

    Returns ``None`` when no path is configured or the file is not found.
    """
    requested = str(cfg.get("checkpoint_path", "") or "").strip()
    if not requested:
        return None
    candidates = []
    if os.path.isabs(requested):
        candidates.append(requested)
    else:
        candidates.append(os.path.join(os.getcwd(), requested))
        candidates.append(os.path.join(SCRIPT_DIR, requested))
        candidates.append(os.path.join(SCRIPT_DIR, os.path.basename(requested)))
    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            if os.path.exists(path):
                return path
    return None


def _load_td_npse_checkpoint_for_eval(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kwargs = dict(checkpoint["model_kwargs"])
    model = TDNPSE_Model(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    train_time_sec = float(checkpoint.get("train_time_sec", 0.0))
    epochs_ran = int(checkpoint.get("epochs_ran", 0))
    return model, checkpoint, train_time_sec, epochs_ran


def _load_nle_checkpoint_for_eval(checkpoint_path, device):
    """Load an NLE checkpoint trained by ``run_nle_singleobs_train.py``."""
    model, checkpoint = NLESBIOfficialFactorized.load_checkpoint(
        checkpoint_path, device=str(device)
    )
    train_time_sec = float(checkpoint.get("train_time_sec", 0.0))
    epochs_ran = int(checkpoint.get("epochs_ran", 0))
    return model, checkpoint, train_time_sec, epochs_ran


def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", lr=None, *, n_obs):
    """Train an SMMD or MMD generator with the corresponding MMD loss.

    Args:
        n_obs: Number of observations per dataset.  Sets the kernel bandwidth
            scale (``[10, 5] / n_obs`` for MMD, ``1 / n_obs`` for SMMD), which
            matches the Bernstein-von Mises posterior contraction rate
            (posterior std ~ ``1 / sqrt(n_obs)``).
    """
    if lr is None:
        lr = LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, epochs)

    model.to(device)
    model.train()

    # Verify device placement
    param_device = next(model.parameters()).device
    print(f"Model {model_type.upper()} initialized on: {param_device}")

    loss_history = []

    print(f"Starting training ({model_type})...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)

            optimizer.zero_grad()

            with torch.enable_grad():
                z = torch.randn(x_batch.size(0), M, model.d, device=device)

                theta_fake = model(x_batch, z)

                if model_type == "smmd":
                    loss = sliced_mmd_loss(
                        theta_batch, theta_fake, num_slices=L, n_obs=n_obs
                    )
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_obs=n_obs)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                if hasattr(model, "G"):
                    l1_loss = 0.0
                    for param in model.G.parameters():
                        l1_loss = l1_loss + torch.sum(torch.abs(param))
                    loss = loss + L1_LAMBDA * l1_loss

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    return loss_history, training_time


def train_bayesflow(
    train_loader,
    epochs,
    device,
    summary_dim=10,
    learning_rate=None,
    weight_decay=1e-4,
):
    """Train BayesFlow model with AdamW + CosineDecay (matching SMMD/MMD setup)."""
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")

    if learning_rate is None:
        learning_rate = LEARNING_RATE
    print("Starting training (BayesFlow with AdamW + CosineDecay)...")
    print(f"BayesFlow optimizer: AdamW, initial_lr={float(learning_rate):.6g}, "
          f"weight_decay={float(weight_decay):.6g}")
    with _global_autograd(False):
        # Build model
        amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=summary_dim)

        # Manual build (one forward pass) to initialize weights
        try:
            first_x, first_theta = next(iter(train_loader))
            dummy_dict = {
                "inference_variables": first_theta.float(),
                "summary_variables": first_x.float(),
            }
            if hasattr(amortized_posterior, "adapter"):
                _ = amortized_posterior.adapter(dummy_dict)
            _ = amortized_posterior.log_prob(dummy_dict)
            print("BayesFlow model built successfully.")
        except Exception as e:
            print(f"Manual build warning: {e}")

        # AdamW + CosineDecay (per-step, matching SMMD/MMD's cosine annealing style)
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(learning_rate),
            decay_steps=total_steps,
            alpha=0.0,  # eta_min / initial_lr = 0, matches SMMD/MMD's eta_min=0
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=float(weight_decay),
        )
        amortized_posterior.compile(optimizer=optimizer)

        loss_history = []
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for x_batch, theta_batch in train_loader:
                batch_dict = {
                    "inference_variables": theta_batch.float(),
                    "summary_variables": x_batch.float(),
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
                    current_lr = float(optimizer.learning_rate)
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f}s")
    return amortized_posterior, loss_history, training_time


def plot_loss(loss_history, title="Training Loss", round_id=0):
    output_dir = f"{RESULTS_BASE}/plots/round_{round_id}/loss"
    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.lower().replace(" ", "_")

    if isinstance(loss_history, dict):
        train_loss = list(loss_history.get("training_loss", []) or [])
        val_loss = list(loss_history.get("validation_loss", []) or [])
        best_val = list(loss_history.get("best_validation_loss", []) or [])

        if not train_loss and not val_loss:
            return

        max_len = max(len(train_loss), len(val_loss), 1)
        epochs = np.arange(1, max_len + 1)

        plt.figure(figsize=(10, 5))
        if train_loss:
            plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="train")
        if val_loss:
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label="validation")
        if best_val:
            plt.axhline(
                best_val[-1], color="red", linestyle="--", label="best_validation"
            )

        plt.title(f"{title} (Round {round_id})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/{safe_title}.png")
        plt.close()

        loss_df = pd.DataFrame({"epoch": epochs})
        if train_loss:
            loss_df["training_loss"] = pd.Series(train_loss, dtype=float)
        if val_loss:
            loss_df["validation_loss"] = pd.Series(val_loss, dtype=float)
        if best_val:
            loss_df["best_validation_loss_final"] = float(best_val[-1])
        loss_df.to_csv(f"{output_dir}/{safe_title}.csv", index=False)
        return

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(f"{title} (Round {round_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{safe_title}.png")
    plt.close()


def plot_posterior(theta_samples, theta_true, output_dir, filename="posterior.png"):
    """
    Plots the marginal posterior for each of the 5 dimensions in a single row.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_params = theta_samples.shape[1]
    # Ensure we have at least 5 dimensions or plot all available
    cols = num_params

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    # Create DataFrame for easier plotting with seaborn
    df = pd.DataFrame(theta_samples, columns=[f"theta{i+1}" for i in range(num_params)])

    for i in range(cols):
        ax = axes[i]
        param_name = f"theta{i+1}"

        # Plot KDE
        sns.kdeplot(
            data=df, x=param_name, fill=True, alpha=0.5, ax=ax, label="Posterior"
        )

        # Plot True Parameter
        if i < len(theta_true):
            ax.axvline(
                x=theta_true[i],
                color="red",
                linestyle="--",
                linewidth=2,
                label="True Parameter",
            )

        ax.set_title(f"Marginal {param_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_combined_posteriors(
    all_model_samples, theta_true, output_dir, filename="combined_posteriors.png"
):
    """
    Plots posteriors from multiple models on one figure.
    all_model_samples: dict {model_name: samples}
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_model_samples:
        return

    # Style settings
    sns.set_theme(
        style="whitegrid", font_scale=1.5
    )  # Switch to whitegrid for cleaner look

    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]

    # X limits
    x_limits = [
        [0.5, 1.5],  # theta_1
        [0.5, 1.5],  # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]

    # Colors
    colors = {
        "SMMD": "#DB1218",  # Red
        "MMD": "#1083CA",  # Blue
        "BAYESFLOW": "#49C926",  # Green
        "DNNABC": "#8315dd",  # Orange/Purple
        "NPE": "#f19327",  # Use W2 color for NPE
        "NLE": "#00A896",  # Teal
        "TD-NPSE": "#6E44FF",  # Indigo
        "TRUEPOSTERIOR": "#363336",  # Dark Grey for True Posterior
        "TRUE": "#FA0101",  # Red for True Value line
    }

    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))  # Slightly reduced height
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)

    if cols == 1:
        axes = [axes]

    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        # Plot models
        for model_name, samples in all_model_samples.items():
            if samples is None:
                continue

            label_name = model_name.upper().replace("_", "-")

            # Determine color
            c = "#333333"
            if "SMMD" in label_name:
                c = colors["SMMD"]
            elif "MMD" in label_name:
                c = colors["MMD"]
            elif "BAYESFLOW" in label_name:
                c = colors["BAYESFLOW"]
            elif "DNN" in label_name:
                c = colors["DNNABC"]
            elif "NLE" in label_name:
                c = colors["NLE"]
            elif "TD-NPSE" in label_name:
                c = colors["TD-NPSE"]
            elif "NPE" in label_name or "SNPE" in label_name:
                c = colors["NPE"]
            elif "TRUEPOSTERIOR" in label_name:
                c = colors["TRUEPOSTERIOR"]

            # Plot
            if "TRUEPOSTERIOR" in label_name:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label="True Posterior",
                    ax=ax,
                    color=c,
                    fill=False,
                    linestyle="-.",
                    linewidth=4.0,
                )
            else:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=label_name,
                    ax=ax,
                    color=c,
                    fill=True,
                    alpha=0.15,
                    linewidth=4.5,
                )


        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(
            axis="both", which="major", labelsize=tick_labelsize, width=2, length=6
        )

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.set_xlim(x_limits[i])

        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    # Legend handling
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Define desired order
    order = [
        "SMMD",
        "MMD",
        "BAYESFLOW",
        "DNNABC",
        "NLE",
        "TD-NPSE",
        "NPE",
        "True Posterior",
    ]

    # Filter and sort
    handles_sorted = []
    labels_sorted = []
    for l in order:
        # Match legend entries by label (exact match or substring).
        for k in by_label.keys():
            if l == "True Posterior" and k == "True Posterior":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l == "True Value" and k == "True Value":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif (
                l in k and "True Posterior" not in k and "True Value" not in k
            ):  # Avoid double matching
                if k not in labels_sorted:
                    handles_sorted.append(by_label[k])
                    labels_sorted.append(k)

    # If the specific order logic misses something, just use what's available
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels

    fig.legend(
        handles_sorted,
        labels_sorted,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),  # Adjusted for bottom margin
        ncol=min(len(handles_sorted), 4),  # Max 4 columns
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5,
    )

    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches="tight")
    plt.close()


def plot_refinement_comparison(
    samples_dict, theta_true, output_dir, model_name, true_posterior_samples=None
):
    """
    Plots Amortized vs Refined for a single model.
    samples_dict: { 'Amortized': samples, 'Refined': samples }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_refinement_comparison.png"

    # Check if we have samples
    valid_samples = [s for s in samples_dict.values() if s is not None]
    if not valid_samples:
        return

    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)

    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]

    # X limits
    x_limits = [
        [0.5, 1.5],  # theta_1
        [0.5, 1.5],  # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]

    colors = {
        "SMMD (Refined)": "#FF3A20",  # Bright Red
        "SMMD (Baseline)": "#DB1218",  # Dark Red
        "MMD (Refined)": "#3FA7D6",  # Bright Blue
        "MMD (Baseline)": "#1083CA",  # Dark Blue
        "BayesFlow (Refined)": "#59CD90",  # Bright Green
        "BayesFlow (Baseline)": "#49C926",  # Dark Green
        "True Posterior": "#363336",
        "True Value": "#FA0101",
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

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        # 1. Plot True Posterior first if available
        if true_posterior_samples is not None:
            sns.kdeplot(
                x=true_posterior_samples[:, dim_idx],
                ax=ax,
                color=colors["True Posterior"],
                fill=False,
                linestyle="-.",
                linewidth=4.0,
                label="True Posterior",
            )

        # 2. Plot Amortized/Refined
        for label, samples in samples_dict.items():
            if samples is None:
                continue

            # Determine color
            # label is usually "Amortized" or "Refined"
            # model_name is "smmd", "mmd", "bayesflow"

            full_label = (
                f"{model_name} ({'Baseline' if label == 'Amortized' else label})"
            )
            # Fix case sensitivity
            if model_name.lower() == "smmd":
                c_key = f"SMMD ({'Baseline' if label == 'Amortized' else 'Refined'})"
            elif model_name.lower() == "mmd":
                c_key = f"MMD ({'Baseline' if label == 'Amortized' else 'Refined'})"
            elif model_name.lower() == "bayesflow":
                c_key = (
                    f"BayesFlow ({'Baseline' if label == 'Amortized' else 'Refined'})"
                )
            else:
                c_key = None

            c = colors.get(c_key, "#333333")

            # Style
            if label == "Amortized":
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_name.upper()} (Amortized)",
                    ax=ax,
                    color=c,
                    fill=False,
                    linestyle="--",
                    linewidth=3.5,
                )
            else:
                sns.kdeplot(
                    x=samples[:, dim_idx],
                    label=f"{model_name.upper()} (Refined)",
                    ax=ax,
                    color=c,
                    fill=True,
                    alpha=0.15,
                    linewidth=4.5,
                )


        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(
            axis="both", which="major", labelsize=tick_labelsize, width=2, length=6
        )

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.set_xlim(x_limits[i])

        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))

    # Sort order
    # Refined, Amortized, True Posterior, True Value
    sorted_handles = []
    sorted_labels = []

    possible_keys = [
        f"{model_name.upper()} (Refined)",
        f"{model_name.upper()} (Amortized)",
        "True Posterior",
    ]

    for k in possible_keys:
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


def plot_all_refinement_comparisons(
    all_results_dict, theta_true, output_dir, true_posterior_samples=None
):
    """
    Plots Amortized vs Refined for ALL models (SMMD, MMD, BayesFlow) on one figure.
    all_results_dict: { model_name: {'Amortized': samples, 'Refined': samples} }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = "all_models_refinement_comparison.png"

    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)

    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]

    # X limits
    x_limits = [
        [0.5, 1.5],  # theta_1
        [0.5, 1.5],  # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]

    colors = {
        "SMMD (Refined)": "#FF3A20",  # Bright Red
        "SMMD (Baseline)": "#DB1218",  # Dark Red
        "MMD (Refined)": "#3FA7D6",  # Bright Blue
        "MMD (Baseline)": "#1083CA",  # Dark Blue
        "BayesFlow (Refined)": "#59CD90",  # Bright Green
        "BayesFlow (Baseline)": "#49C926",  # Dark Green
        "True Posterior": "#363336",
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

    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]

        # 1. Plot True Posterior first
        if true_posterior_samples is not None:
            sns.kdeplot(
                x=true_posterior_samples[:, dim_idx],
                ax=ax,
                color=colors["True Posterior"],
                fill=False,
                linestyle="-.",
                linewidth=4.0,
                label="True Posterior",
            )

        # 2. Plot each model's Amortized and Refined
        # We want to plot all models that have refinement info
        for model_name, result_dict in all_results_dict.items():
            if result_dict is None:
                continue

            # Check for Amortized
            amortized = result_dict.get("Amortized")
            refined = result_dict.get("Refined")

            # Key lookup
            # Fix case sensitivity
            if model_name.lower() == "smmd":
                c_key_base = f"SMMD (Baseline)"
                c_key_ref = f"SMMD (Refined)"
                m_label = "SMMD"
            elif model_name.lower() == "mmd":
                c_key_base = f"MMD (Baseline)"
                c_key_ref = f"MMD (Refined)"
                m_label = "MMD"
            elif model_name.lower() == "bayesflow":
                c_key_base = f"BayesFlow (Baseline)"
                c_key_ref = f"BayesFlow (Refined)"
                m_label = "BayesFlow"
            else:
                continue  # Skip non-refinable models like NPE/DNNABC for this specific plot if not in color dict

            # Plot Amortized
            if amortized is not None:
                c = colors.get(c_key_base, "#333333")
                sns.kdeplot(
                    x=amortized[:, dim_idx],
                    label=f"{m_label} (Amortized)",
                    ax=ax,
                    color=c,
                    fill=False,
                    linestyle="--",
                    linewidth=3.5,
                )

            # Plot Refined
            if refined is not None:
                c = colors.get(c_key_ref, "#333333")
                sns.kdeplot(
                    x=refined[:, dim_idx],
                    label=f"{m_label} (Refined)",
                    ax=ax,
                    color=c,
                    fill=True,
                    alpha=0.15,
                    linewidth=4.5,
                )

        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight="bold", pad=15)
        ax.tick_params(
            axis="both", which="major", labelsize=tick_labelsize, width=2, length=6
        )

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.set_xlim(x_limits[i])

        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Desired order:
    # SMMD(Ref), SMMD(Base), MMD(Ref), MMD(Base), BF(Ref), BF(Base), TruePost
    order = [
        "SMMD (Refined)",
        "SMMD (Amortized)",
        "MMD (Refined)",
        "MMD (Amortized)",
        "BayesFlow (Refined)",
        "BayesFlow (Amortized)",
        "True Posterior",
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


def format_time(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_single_experiment(
    model_type,
    task,
    train_loader,
    theta_true,
    x_obs,
    round_id,
    epochs=None,
    device=DEVICE,
    n_obs=n,
    initial_training_data=None,
    true_posterior_samples=None,
):
    """
    Runs a single experiment for a specific model type.
    """
    if model_type != "bayesflow":
        torch.set_grad_enabled(True)
    resolved_model_cfg = resolve_model_config(model_type)
    model_cfg = {} if resolved_model_cfg is None else dict(resolved_model_cfg)
    if epochs is None:
        epochs = model_cfg.get("epochs", 30)
    print(
        f"\n=== Running Experiment for {model_type.upper()} (Round {round_id}, Epochs={epochs}, n={n_obs}) ==="
    )

    # 1. Model Initialization & Training
    model = None
    loss_history = []
    training_time = 0.0
    refinement_time = 0.0
    total_time = 0.0  # For DNNABC
    td_npse_used_pretrained = False
    nle_used_pretrained = False

    t_start_total = time.time()

    if model_type == "smmd":
        summary_dim = model_cfg.get("summary_dim", 10)
        lr = model_cfg.get("learning_rate", LEARNING_RATE)
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, device, model_type="smmd", lr=lr, n_obs=n_obs
        )
    elif model_type == "mmd":
        summary_dim = model_cfg.get("summary_dim", 10)
        lr = model_cfg.get("learning_rate", LEARNING_RATE)
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, device, model_type="mmd", lr=lr, n_obs=n_obs
        )
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            # On MPS, Keras hits linalg_qr errors → use CPU.
            # On CUDA / CPU, let Keras + Torch backend manage device.
            if torch.cuda.is_available():
                bf_device = torch.device("cuda")
                print(f"✅ BayesFlow using CUDA: {torch.cuda.get_device_name(0)}")
            else:
                bf_device = torch.device("cpu")
                print("ℹ️ BayesFlow using CPU (no CUDA, or MPS not supported by Keras).")
            summary_dim = model_cfg.get("summary_dim", 10)
            lr = model_cfg.get("learning_rate", LEARNING_RATE)
            wd = model_cfg.get("weight_decay", 1e-4)
            model, loss_history, training_time = train_bayesflow(
                train_loader,
                epochs,
                bf_device,
                summary_dim=summary_dim,
                learning_rate=lr,
                weight_decay=wd,
            )
        else:
            print("BayesFlow unavailable, skipping.")
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x)
        loss_history = train_dnnabc(model, train_loader, epochs, device)
    elif model_type == "nle":
        nle_cfg = MODELS_CONFIG.get(model_type, {})
        nle_eval_mode = str(nle_cfg.get("eval_mode", "reload_checkpoint")).lower()
        if nle_eval_mode == "reload_checkpoint":
            checkpoint_path = _resolve_checkpoint_path(nle_cfg)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "NLE eval_mode='reload_checkpoint' but checkpoint_path was not found. "
                    f"Configured path: {nle_cfg.get('checkpoint_path')!r}"
                )
            print(f"Reloading NLE checkpoint for evaluation: {checkpoint_path}")
            model, checkpoint, training_time, checkpoint_epochs = _load_nle_checkpoint_for_eval(
                checkpoint_path,
                device,
            )
            nle_used_pretrained = True
            loss_history = []
            ckpt_prior_type = str(
                checkpoint.get("model_kwargs", {}).get("prior_type", "unknown")
            )
            if ckpt_prior_type != str(task.prior_type):
                print(
                    f"Warning: NLE checkpoint prior_type='{ckpt_prior_type}' differs from "
                    f"current task prior_type='{task.prior_type}'."
                )
            print(
                f"Loaded NLE checkpoint trained for {checkpoint_epochs} epochs "
                f"(recorded training_time={training_time:.2f}s)."
            )
        elif nle_eval_mode == "train_from_scratch":
            lr = nle_cfg.get("learning_rate", 1e-4)
            if initial_training_data is None:
                raise ValueError(
                    "NLE requires initial_training_data=(x_train, theta_train)."
                )
            x_train_np, theta_train_np = initial_training_data
            nle_device = nle_cfg.get("device", str(device))
            print(f"NLE training/inference device request: {nle_device}")
            model = NLESBIOfficialFactorized(
                theta_dim=d,
                x_dim=d_x,
                prior_lower=task.lower,
                prior_upper=task.upper,
                prior_type=task.prior_type,
                hidden_features=nle_cfg.get("hidden_features", 50),
                num_transforms=nle_cfg.get("num_transforms", 5),
                num_blocks=nle_cfg.get("num_blocks", 2),
                z_score_theta=nle_cfg.get("z_score_theta", "independent"),
                z_score_x=nle_cfg.get("z_score_x", "independent"),
                observations_per_theta=nle_cfg.get("observations_per_theta", 10),
                mcmc_method=nle_cfg.get("mcmc_method", "slice_np_vectorized"),
                mcmc_chains=nle_cfg.get("mcmc_chains", 64),
                mcmc_chain_batch_size=nle_cfg.get("mcmc_chain_batch_size", None),
                mcmc_warmup_steps=nle_cfg.get("mcmc_warmup_steps", 200),
                mcmc_thin=nle_cfg.get("mcmc_thin", 1),
                slice_width_mode=nle_cfg.get("slice_width_mode", "constant"),
                slice_init_width=nle_cfg.get("slice_init_width", 0.01),
                slice_reference_n=nle_cfg.get("slice_reference_n", 50.0),
                seed=nle_cfg.get("seed", 0),
                device=nle_device,
            )
            t_train_start = time.time()
            train_summary = model.fit(
                theta_train=theta_train_np,
                x_train=x_train_np,
                max_num_epochs=epochs,
                learning_rate=lr,
                training_batch_size=nle_cfg.get("training_batch_size", BATCH_SIZE),
                validation_fraction=nle_cfg.get("validation_split", 0.05),
                stop_after_epochs=nle_cfg.get("stop_after_epochs", 20),
            )
            training_time = time.time() - t_train_start
            loss_history = train_summary.sbi_summary
        else:
            raise ValueError(
                f"Unknown NLE eval_mode='{nle_eval_mode}'. "
                "Expected 'reload_checkpoint' or 'train_from_scratch'."
            )
    elif model_type == "td_npse":
        td_cfg = MODELS_CONFIG.get(model_type, {})
        td_eval_mode = str(td_cfg.get("eval_mode", "reload_checkpoint")).lower()
        if td_eval_mode == "reload_checkpoint":
            checkpoint_path = _resolve_checkpoint_path(td_cfg)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "TD-NPSE eval_mode='reload_checkpoint' but checkpoint_path was not found. "
                    f"Configured path: {td_cfg.get('checkpoint_path')!r}"
                )
            print(f"Reloading TD-NPSE checkpoint for evaluation: {checkpoint_path}")
            model, checkpoint, training_time, checkpoint_epochs = _load_td_npse_checkpoint_for_eval(
                checkpoint_path,
                device,
            )
            td_npse_used_pretrained = True
            loss_history = []
            ckpt_prior_type = str(checkpoint.get("model_kwargs", {}).get("prior_type", "unknown"))
            if ckpt_prior_type != str(task.prior_type):
                print(
                    f"Warning: TD-NPSE checkpoint prior_type='{ckpt_prior_type}' differs from "
                    f"current task prior_type='{task.prior_type}'."
                )
            print(
                f"Loaded TD-NPSE checkpoint trained for {checkpoint_epochs} epochs "
                f"(recorded training_time={training_time:.2f}s)."
            )
        elif td_eval_mode == "train_from_scratch":
            lr = td_cfg.get("learning_rate", LEARNING_RATE)
            if initial_training_data is None:
                raise ValueError(
                    "TD-NPSE requires initial_training_data=(x_train, theta_train)."
                )
            x_train_np, theta_train_np = initial_training_data
            theta_mean = theta_train_np.mean(axis=0).astype(np.float32)
            theta_std = np.clip(theta_train_np.std(axis=0).astype(np.float32), 1e-6, None)
            x_train_flat = x_train_np.reshape(-1, x_train_np.shape[-1])
            x_mean = x_train_flat.mean(axis=0).astype(np.float32)
            x_std = np.clip(x_train_flat.std(axis=0).astype(np.float32), 1e-6, None)

            model_kwargs = dict(
                theta_dim=d,
                x_dim=d_x,
                hidden_dim=td_cfg.get("hidden_dim", 256),
                num_layers=td_cfg.get("num_layers", 3),
                hidden_features=td_cfg.get("hidden_features", None),
                time_frequencies=td_cfg.get("time_frequencies", 3),
                num_diffusion_steps=td_cfg.get("num_diffusion_steps", 100),
                sample_steps=td_cfg.get("sample_steps", 1000),
                sampler_type=td_cfg.get("sampler_type", "ddim"),
                cov_steps=td_cfg.get("cov_steps", 100),
                cov_samples=td_cfg.get("cov_samples", 256),
                ddim_eta=td_cfg.get("ddim_eta", 1.0),
                cov_reg=td_cfg.get("cov_reg", 1e-3),
                covariance_scale=td_cfg.get("covariance_scale", 1.0),
                cov_mode=td_cfg.get("cov_mode", "GAUSS"),
                clip_denoised=td_cfg.get("clip_denoised", True),
                prior_lower=task.lower,
                prior_upper=task.upper,
                prior_type=task.prior_type,
                theta_mean=theta_mean,
                theta_std=theta_std,
                x_mean=x_mean,
                x_std=x_std,
                uniform_prior_scale=td_cfg.get("uniform_prior_scale", 1.0),
                tall_score_mode=td_cfg.get("tall_score_mode", "gauss_refine"),
                symmetry_sign_flip_dims=td_cfg.get("symmetry_sign_flip_dims", None),
                symmetry_mode_sampling_dims=td_cfg.get("symmetry_mode_sampling_dims", None),
                covariance_symmetry_dims=td_cfg.get("covariance_symmetry_dims", None),
            )
            model = TDNPSE_Model(**model_kwargs)
            td_dataset = FullSingleObservationDataset(theta_train_np, x_train_np)
            td_loader = DataLoader(
                td_dataset,
                batch_size=td_cfg.get("training_batch_size", BATCH_SIZE),
                shuffle=True,
            )
            t_train_start = time.time()
            loss_history = train_td_npse(
                model,
                td_loader,
                epochs,
                device,
                learning_rate=lr,
                lr_schedule=td_cfg.get("lr_schedule", "none"),
                eta_min=td_cfg.get("eta_min", 0.0),
                validation_split=td_cfg.get("validation_split", 0.0),
                early_stopping=td_cfg.get("early_stopping", False),
                patience=td_cfg.get("patience", 20000),
                min_nb_epochs=td_cfg.get("min_nb_epochs", 100),
            )
            training_time = time.time() - t_train_start
        else:
            raise ValueError(
                f"Unknown TD-NPSE eval_mode='{td_eval_mode}'. "
                "Expected 'reload_checkpoint' or 'train_from_scratch'."
            )
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        sbi_rounds = model_cfg.get("sbi_rounds", 5)
        sims_per_round = model_cfg.get("sims_per_round", 1000)
        posterior_samples = run_sbi_model(
            model_type=model_type,
            train_loader=None,
            x_obs=x_obs,
            theta_true=theta_true,
            task=task,
            device=str(device),
            num_rounds=sbi_rounds,
            sims_per_round=sims_per_round,
            max_epochs=epochs,
            n_samples=N_SAMPLES_POSTERIOR,
            validation_fraction=model_cfg.get("validation_fraction", 0.1),
            stop_after_epochs=model_cfg.get("stop_after_epochs", 20),
            learning_rate=model_cfg.get("learning_rate", 5e-4),
            training_batch_size=model_cfg.get("training_batch_size", 200),
            show_train_summary=model_cfg.get("show_train_summary", False),
            mcmc_chains=model_cfg.get("mcmc_chains", 20),
            initial_training_data=initial_training_data,
        )
        loss_history = []

    if loss_history:
        plot_loss(loss_history, title=f"{model_type.upper()} Loss", round_id=round_id)

    # 2. Model Inference (Amortized)
    print(f"--- Amortized Inference ({model_type}) ---")

    # Sample from model
    n_samples = N_SAMPLES_POSTERIOR

    if model_type == "nle":
        posterior_samples = model.sample_posterior(x_obs, n_samples=n_samples)
    elif model_type == "td_npse":
        posterior_samples = model.sample_posterior(x_obs, n_samples=n_samples)
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        # Already sampled in training step
        pass
    elif model_type == "dnnabc":
        dnn_cfg = MODELS_CONFIG.get("dnnabc", {})
        n_pool = dnn_cfg.get("n_pool", 100000)
        abc_batch_size = dnn_cfg.get("abc_batch_size", 10000)
        posterior_samples = abc_rejection_sampling(
            model,
            x_obs,
            task,
            n_samples=n_samples,
            n_pool=n_pool,
            batch_size=abc_batch_size,
            device=device,
        )
    elif hasattr(model, "sample_posterior"):
        posterior_samples = model.sample_posterior(x_obs, n_samples)
    elif model_type == "bayesflow":
        # Prepare conditions for sampling
        # BayesFlow expects conditions as dict (via adapter)
        # IMPORTANT: Adapter requires CPU/Numpy data
        if isinstance(x_obs, torch.Tensor):
            x_obs_cpu = x_obs.detach().cpu().numpy()
        else:
            x_obs_cpu = np.asarray(x_obs)

        # Add batch dim if needed
        if x_obs_cpu.ndim == 2:
            x_obs_cpu = x_obs_cpu[np.newaxis, ...]

        conditions_dict = {"summary_variables": x_obs_cpu}

        # Sample
        posterior_samples = model.sample(
            conditions=conditions_dict, num_samples=n_samples
        )

        # Output might be dict
        if isinstance(posterior_samples, dict):
            posterior_samples = posterior_samples["inference_variables"]

        # Shape: (1, n_samples, d) -> (n_samples, d)
        posterior_samples = posterior_samples.reshape(-1, d)
    else:
        # Fallback (should typically not happen if wrappers are correct)
        print("Model does not support sample_posterior, skipping inference.")
        return None

    print(f"Posterior Mean: {np.mean(posterior_samples, axis=0)}")

    # Plot Amortized Posterior
    plot_dir = f"{RESULTS_BASE}/plots/round_{round_id}/{model_type}"
    plot_posterior(posterior_samples, theta_true, plot_dir, "amortized_posterior.png")

    # 3. Local Refinement
    refined_samples_flat = None
    refinement_time = 0.0

    if model_type in [
        "dnnabc",
        "snpe_a",
        "npe",
        "snpe",
        "snpe_c",
        "nle",
        "td_npse",
    ]:
        print(f"--- Skipping Local Refinement for {model_type} ---")
        refined_samples_flat = posterior_samples
        eval_wall_time = time.time() - t_start_total
        total_time = (
            training_time + eval_wall_time
            if (td_npse_used_pretrained or nle_used_pretrained)
            else eval_wall_time
        )
    else:
        print(f"--- Local Refinement ({model_type}) ---")

        t_refine_start = time.time()

        # Refinement parameters
        n_chains = REFINE_CONFIG.get("n_chains", 2000)
        burn_in = REFINE_CONFIG.get("burn_in", 99)
        thin = REFINE_CONFIG.get("thin", 1)
        nsims = REFINE_CONFIG.get("nsims", 50)
        epsilon = REFINE_CONFIG.get("epsilon", None)
        proposal_std = REFINE_CONFIG.get("proposal_std", 0.2)
        quantile_level = REFINE_CONFIG.get("quantile_level", 0.01)
        bandwidth_n_samples = REFINE_CONFIG.get("bandwidth_n_samples", 20000)

        refined_samples = refine_posterior(
            model,
            x_obs,
            task,
            n_chains=n_chains,
            n_samples=1,  # Take the last sample
            burn_in=burn_in,
            thin=thin,
            nsims=nsims,
            epsilon=epsilon,
            proposal_std=proposal_std,
            quantile_level=quantile_level,
            bandwidth_n_samples=bandwidth_n_samples,
            device=str(device),
        )

        refinement_time = time.time() - t_refine_start
        print(f"Refinement finished in {refinement_time:.2f}s")

        # Reshape refined samples: (n_chains * n_samples, d)
        refined_samples_flat = refined_samples.reshape(-1, d)

        print(f"Refined Mean: {np.mean(refined_samples_flat, axis=0)}")

        # Plot Refined Posterior
        plot_posterior(
            refined_samples_flat, theta_true, plot_dir, "refined_posterior.png"
        )

        # Plot Comparison (Amortized vs Refined)
        samples_comp = {"Amortized": posterior_samples, "Refined": refined_samples_flat}
        plot_refinement_comparison(
            samples_comp,
            theta_true,
            plot_dir,
            model_type,
            true_posterior_samples=true_posterior_samples,
        )

        total_time = (
            training_time + refinement_time
        )  # Though for these we log separately

    return {
        "model": model,
        "loss_history": loss_history,
        "amortized_samples": posterior_samples,
        "refined_samples": refined_samples_flat,
        "training_time": training_time,
        "refinement_time": refinement_time,
        "total_time": total_time,
    }


def save_model(model, model_type, round_id):
    """Saves the model state."""
    if not SAVE_MODEL_PARAMS:
        return
    save_dir = f"saved_models_{n}{RESULTS_SUFFIX}/{model_type}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/round_{round_id}.pt"

    if model_type == "bayesflow":
        try:
            model.save(f"{save_dir}/round_{round_id}.keras")
            print(f"Saved BayesFlow model to {save_dir}/round_{round_id}.keras")
        except Exception as e:
            print(f"Failed to save BayesFlow model (full): {e}")
            try:
                model.save_weights(f"{save_dir}/round_{round_id}.weights.h5")
                print(
                    f"Saved BayesFlow weights to {save_dir}/round_{round_id}.weights.h5"
                )
            except Exception as e2:
                print(f"Failed to save BayesFlow weights: {e2}")

                try:
                    if hasattr(model, "summary_network"):
                        model.summary_network.save_weights(
                            f"{save_dir}/round_{round_id}_summary.weights.h5"
                        )
                        print(
                            f"Saved Summary Network weights to {save_dir}/round_{round_id}_summary.weights.h5"
                        )
                except Exception as e3:
                    print(f"Failed to save Summary Network weights: {e3}")

                try:
                    if hasattr(model, "inference_network"):
                        dummy_x = torch.zeros((1, n, d_x), dtype=torch.float32)
                        x_emb = model.summary_network(dummy_x)
                        dummy_theta = torch.zeros((1, d), dtype=torch.float32)
                        if hasattr(model.inference_network, "log_prob"):
                            _ = model.inference_network.log_prob(
                                dummy_theta, conditions=x_emb
                            )
                        else:
                            _ = model.inference_network(dummy_theta, conditions=x_emb)
                        torch.save(
                            model.inference_network.state_dict(),
                            f"{save_dir}/round_{round_id}_inference.pt",
                        )
                        print(
                            f"Saved Inference Network state_dict to {save_dir}/round_{round_id}_inference.pt"
                        )
                except Exception as e4:
                    print(f"Failed to save Inference Network state_dict: {e4}")
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename)
        print(f"Saved {model_type} model to {filename}")
    else:
        print(f"Skipping save for {model_type} (not a torch module or bayesflow).")


def main():
    global RESULTS_BASE, n
    # Reproducibility: base seed; each (n_obs, round) combination is re-seeded
    # with ``base_seed + n_obs_idx * NUM_ROUNDS + (round_idx - 1)`` so every
    # round's data generation is deterministic and independently reproducible.
    base_seed = int(CONFIG.get("seed", 0))

    if SAVE_MODEL_PARAMS:
        os.makedirs("saved_models", exist_ok=True)

    print(f"Starting Multi-Round Experiment (Total Rounds: {NUM_ROUNDS})")
    print(f"n_observation settings: {N_OBS_LIST}")

    for n_obs_idx, n_curr in enumerate(N_OBS_LIST):
        n = n_curr
        RESULTS_BASE = os.path.join(_SCRIPT_DIR, f"results_{n_curr}{RESULTS_SUFFIX}")
        os.makedirs(RESULTS_BASE, exist_ok=True)
        _setup_terminal_log(RESULTS_BASE)

        all_results = []
        model_results_table = {}

        print(f"\n******** Running experiments for n_observation = {n_curr} ********")

        for round_idx in range(1, NUM_ROUNDS + 1):
            round_seed = base_seed + n_obs_idx * NUM_ROUNDS + (round_idx - 1)
            np.random.seed(round_seed)
            torch.manual_seed(round_seed)
            torch.cuda.manual_seed_all(round_seed)
            print(f"\n{'=' * 20} ROUND {round_idx}/{NUM_ROUNDS} | seed={round_seed} {'=' * 20}")

            print("=== Step 1: Data Generation ===")
            task = GaussianTask(n=n)
            print(f"Generating training data (size={DATASET_SIZE}, n={n})...")
            theta_train, x_train = generate_dataset(task, n_sims=DATASET_SIZE, n_obs=n)

            dataset = TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(theta_train).float(),
            )
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            print("Generating observation...")
            theta_true, x_obs = task.get_ground_truth()
            print(f"True Params: {theta_true}")

            print("\n=== Step 2: True Posterior Sampling ===")
            true_samples = None
            try:
                X_comp = x_obs[:, 0]
                Y_comp = x_obs[:, 1]
                Z_comp = x_obs[:, 2]

                denom = 1.0 - Z_comp
                denom[np.abs(denom) < 1e-6] = 1e-6

                x_2d = X_comp / denom
                y_2d = Y_comp / denom
                x_obs_2d = np.stack([x_2d, y_2d], axis=-1)

                print(
                    f"Inverted 3D obs to 2D for true posterior sampling. "
                    f"Shape: {x_obs_2d.shape}"
                )

                coeff = TRUE_POSTERIOR_CONFIG.get("proposal_scale_coeff", None)
                if coeff is not None:
                    proposal_scale = float(coeff) / float(np.sqrt(float(n)))
                else:
                    base_scale = float(
                        TRUE_POSTERIOR_CONFIG.get("proposal_scale", 0.06)
                    )
                    ref_n = float(
                        TRUE_POSTERIOR_CONFIG.get(
                            "proposal_scale_reference_n", 25.0
                        )
                    )
                    proposal_scale = base_scale * float(np.sqrt(ref_n / float(n)))
                print(
                    f"True posterior MCMC proposal_scale={proposal_scale:.6g} (n_obs={n})"
                )
                true_samples = run_gaussian_posterior_mcmc(
                    x_obs_2d,
                    task,
                    n_draws=TRUE_POSTERIOR_CONFIG.get("n_draws", 2000),
                    n_tune_chain=TRUE_POSTERIOR_CONFIG.get("n_tune_chain", 1000),
                    chains=TRUE_POSTERIOR_CONFIG.get("chains", 30),
                    proposal_scale=proposal_scale,
                )

                print(f"True Posterior Samples Shape: {true_samples.shape}")
                true_plot_dir = f"{RESULTS_BASE}/plots/round_{round_idx}/true_posterior"
                plot_posterior(true_samples, theta_true, true_plot_dir, "posterior.png")

                true_save_dir = (
                    f"{RESULTS_BASE}/posterior_samples/true_reference/round_{round_idx}"
                )
                os.makedirs(true_save_dir, exist_ok=True)
                np.save(f"{true_save_dir}/samples.npy", true_samples)
                print(f"Saved true posterior samples to {true_save_dir}/samples.npy")
            except Exception as e:
                print(f"True posterior sampling failed: {e}")
                true_samples = None

            print("\n=== Step 3: Run Model Experiments ===")
            round_samples_for_plot = {}
            refinement_samples_for_plot = {}
            if true_samples is not None:
                round_samples_for_plot["TruePosterior"] = true_samples

            current_models_to_run = MODELS_TO_RUN
            if isinstance(current_models_to_run, str):
                current_models_to_run = [current_models_to_run]

            for model_type in current_models_to_run:
                if resolve_model_config(model_type) is None:
                    print(f"Warning: No config for {model_type}, skipping.")
                    continue

                initial_data = (x_train, theta_train)

                result = run_single_experiment(
                    model_type,
                    task,
                    train_loader,
                    theta_true,
                    x_obs,
                    round_id=round_idx,
                    device=DEVICE,
                    n_obs=n,
                    initial_training_data=initial_data,
                    true_posterior_samples=true_samples,
                )

                if result is None:
                    continue

                if result["amortized_samples"] is not None:
                    round_samples_for_plot[model_type] = result["amortized_samples"]

                if (
                    model_type in ["smmd", "mmd", "bayesflow"]
                    and result["refined_samples"] is not None
                ):
                    refinement_samples_for_plot[model_type] = {
                        "Amortized": result["amortized_samples"],
                        "Refined": result["refined_samples"],
                    }

                mmd_amortized = np.nan
                mmd_refined = np.nan
                swd_amortized = np.nan
                swd_refined = np.nan

                if true_samples is not None:
                    swd_scale = float(np.sqrt(float(n)))
                    if result["amortized_samples"] is not None:
                        mmd_amortized = compute_mmd_metric(
                            result["amortized_samples"], true_samples
                        )
                        print(f"MMD ({model_type} Amortized): {mmd_amortized:.4f}")
                        swd_amortized = compute_swd_metric(
                            np.asarray(result["amortized_samples"], dtype=np.float32)
                            * swd_scale,
                            np.asarray(true_samples, dtype=np.float32) * swd_scale,
                            n_projections=int(CONFIG.get("swd_n_projections", 1000)),
                            max_samples=int(CONFIG.get("swd_max_samples", 2000)),
                            seed=round_seed,
                        )
                        print(
                            f"SWD ({model_type} Amortized, scaled by sqrt(n_obs)): {swd_amortized:.4f}"
                        )

                    if (
                        model_type in ["smmd", "mmd", "bayesflow"]
                        and result["refined_samples"] is not None
                    ):
                        mmd_refined = compute_mmd_metric(
                            result["refined_samples"], true_samples
                        )
                        print(f"MMD ({model_type} Refined): {mmd_refined:.4f}")
                        swd_refined = compute_swd_metric(
                            np.asarray(result["refined_samples"], dtype=np.float32)
                            * swd_scale,
                            np.asarray(true_samples, dtype=np.float32) * swd_scale,
                            n_projections=int(CONFIG.get("swd_n_projections", 1000)),
                            max_samples=int(CONFIG.get("swd_max_samples", 2000)),
                            seed=round_seed,
                        )
                        print(
                            f"SWD ({model_type} Refined, scaled by sqrt(n_obs)): {swd_refined:.4f}"
                        )
                else:
                    print("Skipping MMD calculation (no true posterior samples).")

                if SAVE_MODEL_PARAMS:
                    save_model(result["model"], model_type, round_idx)

                record = {
                    "round": round_idx,
                    "seed": round_seed,
                    "model_name": model_type,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "training_time": result["training_time"],
                    "refinement_time": result["refinement_time"],
                    "total_time": result["total_time"],
                    "training_time_hms": format_time(result["training_time"]),
                    "refinement_time_hms": format_time(result["refinement_time"]),
                    "total_time_hms": format_time(result["total_time"]),
                    "mmd_amortized": mmd_amortized,
                    "mmd_refined": mmd_refined,
                    "swd_amortized": swd_amortized,
                    "swd_refined": swd_refined,
                }
                all_results.append(record)

                if model_type not in model_results_table:
                    model_results_table[model_type] = []
                model_results_table[model_type].append(record)

                # Table 1: per-round MMD / sqrt(n)*sW values (amortized and refined).
                pd.DataFrame(all_results).to_csv(
                    f"{RESULTS_BASE}/experiment_results.csv", index=False
                )

            print(f"Generating combined posterior plot for Round {round_idx}...")
            combined_plot_dir = f"{RESULTS_BASE}/plots/round_{round_idx}"
            plot_combined_posteriors(
                round_samples_for_plot,
                theta_true,
                combined_plot_dir,
                "combined_posteriors.png",
            )

            # Plot all-model refinement comparisons; samples were collected in
            # refinement_samples_for_plot during the model loop above.
            print(
                f"Generating all-model refinement comparison plot for Round {round_idx}..."
            )
            # Samples are kept in memory (no per-model .npy files are written
            # to disk anymore) and collected during the model loop above.
            if refinement_samples_for_plot:
                plot_all_refinement_comparisons(
                    refinement_samples_for_plot,
                    theta_true,
                    combined_plot_dir,
                    true_posterior_samples=true_samples,
                )
            else:
                print("No refinement data found for all-model comparison.")

        print("\n=== Experiment Summary ===")
        if not model_results_table:
            print("Warning: No results to summarize.")
        else:
            summary_rows = []

            for model_name, rows in model_results_table.items():
                if not rows:
                    continue

                per_round_rows = []
                for r in rows:
                    per_round_rows.append(
                        {
                            "round": r["round"],
                            "MMD_Amortized": r.get("mmd_amortized", np.nan),
                            "MMD_Refined": r.get("mmd_refined", np.nan),
                            "SWD_Amortized": r.get("swd_amortized", np.nan),
                            "SWD_Refined": r.get("swd_refined", np.nan),
                        }
                    )

                per_round_df = pd.DataFrame(per_round_rows)

                mmd_am_vals = per_round_df["MMD_Amortized"].dropna().values
                mmd_ref_vals = per_round_df["MMD_Refined"].dropna().values
                swd_am_vals = per_round_df["SWD_Amortized"].dropna().values
                swd_ref_vals = per_round_df["SWD_Refined"].dropna().values

                mmd_am_mean = (
                    float(np.mean(mmd_am_vals)) if len(mmd_am_vals) > 0 else np.nan
                )
                mmd_am_median = (
                    float(np.median(mmd_am_vals)) if len(mmd_am_vals) > 0 else np.nan
                )
                mmd_ref_mean = (
                    float(np.mean(mmd_ref_vals)) if len(mmd_ref_vals) > 0 else np.nan
                )
                mmd_ref_median = (
                    float(np.median(mmd_ref_vals)) if len(mmd_ref_vals) > 0 else np.nan
                )
                swd_am_mean = (
                    float(np.mean(swd_am_vals)) if len(swd_am_vals) > 0 else np.nan
                )
                swd_am_median = (
                    float(np.median(swd_am_vals)) if len(swd_am_vals) > 0 else np.nan
                )
                swd_ref_mean = (
                    float(np.mean(swd_ref_vals)) if len(swd_ref_vals) > 0 else np.nan
                )
                swd_ref_median = (
                    float(np.median(swd_ref_vals)) if len(swd_ref_vals) > 0 else np.nan
                )

                summary_rows.append(
                    {
                        "Model": model_name,
                        "MMD_Amortized_Mean": mmd_am_mean,
                        "MMD_Amortized_Median": mmd_am_median,
                        "MMD_Refined_Mean": mmd_ref_mean,
                        "MMD_Refined_Median": mmd_ref_median,
                        "SWD_Amortized_Mean": swd_am_mean,
                        "SWD_Amortized_Median": swd_am_median,
                        "SWD_Refined_Mean": swd_ref_mean,
                        "SWD_Refined_Median": swd_ref_median,
                    }
                )

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                summary_path = f"{RESULTS_BASE}/final_summary.csv"
                # Table 1: per-model summary statistics averaged over replications.
                df_summary.to_csv(summary_path, index=False)
                print(f"Saved final summary to {summary_path}")
            else:
                print("No valid summary rows were generated.")

    print("\nAll Experiments Completed.")


if __name__ == "__main__":
    main()
