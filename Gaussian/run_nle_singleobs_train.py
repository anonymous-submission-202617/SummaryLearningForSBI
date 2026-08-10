"""Train the NLE model for the Gaussian main experiment on single-observation data.

Trains a neural likelihood estimator (MAF-based, as configured in
``config.json``) on single-observation training pairs (one observation per
theta draw) sampled from a uniform prior over [-3, 3]^5, then saves the fitted
estimator to a checkpoint file.  ``run_experiment.py`` loads this checkpoint
in ``eval_mode="reload_checkpoint"`` and only performs posterior inference
during the main experiment rounds (mirroring the TD-NPSE workflow).

The checkpoint stores the full constructor configuration together with the
estimator state dict, so the inference settings (slice-sampler chains, warmup,
thin, ...) are fixed at training time and reproduced exactly on reload.

Example:
    python run_nle_singleobs_train.py --n-train 50000 --epochs 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from data_generation import GaussianTask, generate_dataset
from models.nle import NLESBIOfficialFactorized


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch (and CUDA if present) for reproducibility."""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_select_device(requested: str) -> torch.device:
    """Resolve the requested device string to a torch device."""
    requested = str(requested or "auto").strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Auto-detected CUDA acceleration. Device: {device}")
            return device
        device = torch.device("cpu")
        print(f"Auto-detected CPU fallback. Device: {device}")
        return device

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested)
            print(f"Using requested CUDA device: {device}")
            return device
        print(f"Warning: requested device '{requested}' but CUDA is not available. Falling back to auto.")
        return auto_select_device("auto")

    if requested == "cpu":
        device = torch.device("cpu")
        print(f"Using requested CPU device: {device}")
        return device

    if requested == "mps":
        print("Warning: MPS is not supported by the sbi stack; falling back to CPU.")
        return auto_select_device("cpu")

    print(f"Warning: unsupported device='{requested}'. Expected auto/cpu/cuda[:k]. Falling back to auto.")
    return auto_select_device("auto")


def load_nle_config() -> Dict[str, Any]:
    """Load the NLE model configuration from Gaussian/config.json, if present."""
    config_path = os.path.join(_SCRIPT_DIR, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        full = json.load(f)
    cfg = full.get("models_config", {}).get("nle", {})
    return dict(cfg) if isinstance(cfg, dict) else {}


def build_parser(nle_cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the CLI parser; defaults are taken from the NLE config entry."""

    def cfg_get(key: str, default: Any) -> Any:
        value = nle_cfg.get(key)
        return default if value is None else value

    parser = argparse.ArgumentParser(
        description="Train the Gaussian NLE model on single-observation data "
        "(uniform prior) and save a checkpoint for run_experiment.py."
    )
    parser.add_argument("--device", default=os.environ.get("GAUSSIAN_DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=50000,
                        help="Number of (theta, single-observation) training pairs.")
    parser.add_argument("--train-n-obs", type=int, default=1,
                        help="Number of observations simulated per theta draw.")
    parser.add_argument("--epochs", type=int, default=cfg_get("epochs", 200))
    parser.add_argument("--training-batch-size", type=int,
                        default=cfg_get("training_batch_size", 256))
    parser.add_argument("--learning-rate", type=float,
                        default=cfg_get("learning_rate", 1e-4))
    parser.add_argument("--validation-split", type=float,
                        default=cfg_get("validation_split", 0.05))
    parser.add_argument("--stop-after-epochs", type=int,
                        default=cfg_get("stop_after_epochs", 20))
    parser.add_argument("--hidden-features", type=int,
                        default=cfg_get("hidden_features", 50))
    parser.add_argument("--num-transforms", type=int,
                        default=cfg_get("num_transforms", 5))
    parser.add_argument("--num-blocks", type=int,
                        default=cfg_get("num_blocks", 2))
    parser.add_argument("--z-score-theta", default=cfg_get("z_score_theta", "independent"))
    parser.add_argument("--z-score-x", default=cfg_get("z_score_x", "independent"))
    parser.add_argument("--observations-per-theta", type=int,
                        default=cfg_get("observations_per_theta", -1))
    parser.add_argument("--mcmc-method", default=cfg_get("mcmc_method", "slice_np_vectorized"))
    parser.add_argument("--mcmc-chains", type=int, default=cfg_get("mcmc_chains", 64))
    parser.add_argument("--mcmc-chain-batch-size", type=int,
                        default=int(cfg_get("mcmc_chain_batch_size", 0) or 0),
                        help="0 disables batching (all chains at once).")
    parser.add_argument("--mcmc-warmup-steps", type=int,
                        default=cfg_get("mcmc_warmup_steps", 200))
    parser.add_argument("--mcmc-thin", type=int, default=cfg_get("mcmc_thin", 1))
    parser.add_argument("--slice-width-mode", default=cfg_get("slice_width_mode", "constant"))
    parser.add_argument("--slice-init-width", type=float,
                        default=cfg_get("slice_init_width", 0.01))
    parser.add_argument("--slice-reference-n", type=float,
                        default=cfg_get("slice_reference_n", 50.0))
    parser.add_argument("--checkpoint-path", default=os.path.join("data", "nle_checkpoint.pt"),
                        help="Where to save the trained NLE checkpoint.")
    parser.add_argument("--output-dir", default="results/nle_singleobs_train",
                        help="Directory for artifacts (training data, run config, reload check).")
    parser.add_argument("--validate-n-pairs", type=int, default=1024,
                        help="Number of (theta, x) pairs used for the reload check; 0 skips it.")
    return parser


def _estimator_log_prob(
    model: NLESBIOfficialFactorized,
    theta_pairs: np.ndarray,
    x_pairs: np.ndarray,
) -> np.ndarray:
    """Evaluate the learned likelihood estimator log p(x | theta) at fixed pairs."""
    device = model.sbi_device
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x_pairs, dtype=torch.float32, device=device)
        theta_t = torch.as_tensor(theta_pairs, dtype=torch.float32, device=device)
        log_prob = model.likelihood_estimator.log_prob(x_t.unsqueeze(0), theta_t)[0]
    return log_prob.detach().cpu().numpy().astype(np.float64)


def validate_checkpoint_reload(
    checkpoint_path: Path,
    original_model: NLESBIOfficialFactorized,
    task: GaussianTask,
    n_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    """Reload the checkpoint and compare estimator outputs to the trained model.

    A passed check means the reloaded estimator reproduces the trained
    estimator's log-likelihood up to numerical precision.
    """
    if n_pairs <= 0:
        return {"passed": None, "skipped": True}
    reloaded_model, checkpoint = NLESBIOfficialFactorized.load_checkpoint(
        str(checkpoint_path), device=str(original_model.device)
    )

    rng = np.random.RandomState(int(seed))
    theta_pairs = rng.uniform(-3.0, 3.0, size=(int(n_pairs), task.d)).astype(np.float32)
    x_pairs = np.asarray(
        task.simulator(theta_pairs, n_samples=1, rng=rng), dtype=np.float32
    ).reshape(int(n_pairs), -1)

    lp_original = _estimator_log_prob(original_model, theta_pairs, x_pairs)
    lp_reloaded = _estimator_log_prob(reloaded_model, theta_pairs, x_pairs)

    max_abs_diff = float(np.max(np.abs(lp_original - lp_reloaded)))
    mean_abs_diff = float(np.mean(np.abs(lp_original - lp_reloaded)))
    return {
        "passed": bool(max_abs_diff < 1e-5),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "n_pairs": int(n_pairs),
        "checkpoint_epochs_ran": int(checkpoint.get("epochs_ran", 0)),
        "checkpoint_train_time_sec": float(checkpoint.get("train_time_sec", 0.0)),
    }


def main() -> None:
    nle_cfg = load_nle_config()
    parser = build_parser(nle_cfg)
    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_select_device(args.device)

    train_n_obs = int(args.train_n_obs)
    task = GaussianTask(n=train_n_obs, prior_type="uniform")

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path(_SCRIPT_DIR) / args.checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    output_dir = (Path(_SCRIPT_DIR) / args.output_dir).resolve()
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Generate single-observation training data (uniform prior) ===")
    theta_train, x_train = generate_dataset(
        task, n_sims=int(args.n_train), n_obs=train_n_obs
    )
    np.save(artifacts_dir / "theta_train.npy", theta_train)
    np.save(artifacts_dir / "x_train.npy", x_train)
    print(f"Training data shapes: theta={theta_train.shape}, x={x_train.shape}")

    print("=== Step 2: Train NLE ===")
    mcmc_chain_batch_size = (
        None if int(args.mcmc_chain_batch_size) <= 0 else int(args.mcmc_chain_batch_size)
    )
    model = NLESBIOfficialFactorized(
        theta_dim=task.d,
        x_dim=task.d_x,
        prior_lower=task.lower,
        prior_upper=task.upper,
        prior_type=task.prior_type,
        hidden_features=int(args.hidden_features),
        num_transforms=int(args.num_transforms),
        num_blocks=int(args.num_blocks),
        z_score_theta=str(args.z_score_theta),
        z_score_x=str(args.z_score_x),
        observations_per_theta=int(args.observations_per_theta),
        mcmc_method=str(args.mcmc_method),
        mcmc_chains=int(args.mcmc_chains),
        mcmc_chain_batch_size=mcmc_chain_batch_size,
        mcmc_warmup_steps=int(args.mcmc_warmup_steps),
        mcmc_thin=int(args.mcmc_thin),
        slice_width_mode=str(args.slice_width_mode),
        slice_init_width=float(args.slice_init_width),
        slice_reference_n=float(args.slice_reference_n),
        seed=int(args.seed),
        device=str(device),
    )
    t0 = time.time()
    train_summary = model.fit(
        theta_train=theta_train,
        x_train=x_train,
        max_num_epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        training_batch_size=int(args.training_batch_size),
        validation_fraction=float(args.validation_split),
        stop_after_epochs=int(args.stop_after_epochs),
    )
    train_time_sec = float(time.time() - t0)

    sbi_summary = train_summary.sbi_summary or {}
    epochs_trained = sbi_summary.get("epochs_trained", [])
    epochs_ran = int(epochs_trained[-1]) if epochs_trained else int(args.epochs)
    print(
        f"Training finished: {epochs_ran} epochs, {train_time_sec:.2f}s "
        f"({train_time_sec / max(epochs_ran, 1):.2f}s/epoch)."
    )

    print(f"=== Step 3: Save checkpoint to {checkpoint_path} ===")
    model.save_checkpoint(
        str(checkpoint_path),
        train_time_sec=train_time_sec,
        epochs_ran=epochs_ran,
        extra={
            "n_train": int(args.n_train),
            "train_n_obs": train_n_obs,
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "training_batch_size": int(args.training_batch_size),
            "epochs": int(args.epochs),
            "stop_after_epochs": int(args.stop_after_epochs),
            "validation_split": float(args.validation_split),
            "device": str(device),
        },
    )

    print("=== Step 4: Validate checkpoint reload ===")
    reload_check = validate_checkpoint_reload(
        checkpoint_path,
        model,
        task,
        n_pairs=int(args.validate_n_pairs),
        seed=int(args.seed) + 1,
    )
    print(json.dumps(reload_check, indent=2))

    with open(artifacts_dir / "run_config.json", "w") as f:
        model_kwargs = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in model.get_model_kwargs().items()
        }
        json.dump(
            {
                "n_train": int(args.n_train),
                "train_n_obs": train_n_obs,
                "seed": int(args.seed),
                "model_kwargs": model_kwargs,
                "train_config": {
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "training_batch_size": int(args.training_batch_size),
                    "validation_split": float(args.validation_split),
                    "stop_after_epochs": int(args.stop_after_epochs),
                },
                "train_time_sec": train_time_sec,
                "epochs_ran": epochs_ran,
                "device": str(device),
            },
            f,
            indent=2,
        )
    with open(artifacts_dir / "reload_check.json", "w") as f:
        json.dump(reload_check, f, indent=2)

    print(f"\nDone. Checkpoint saved to: {checkpoint_path}")
    print(
        "Run the Gaussian main experiment with nle eval_mode='reload_checkpoint' "
        "(the default in config.json) to evaluate this model."
    )


if __name__ == "__main__":
    main()
