###############################################################################
# Gaussian data-size dependence study (SMMD vs NPE)
#
# Trains SMMD and NPE across training budgets N = 20,000 / 30,000 / 40,000
# and observation counts n = 10 / 75 / 150 / 225 / 300, evaluated on 25
# test datasets.
#
# Reproduces Figure 2 of the manuscript: sqrt(n) * sW between estimated and
# ground-truth posteriors as a function of n under different training sizes.
#
# Outputs: results_datasize_dependence/ (see README.md).
###############################################################################

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sbi_common.config_utils import cli_config_arg, resolve_config_path  # noqa: E402
from data_generation import TRUE_PARAMS, GaussianTask, d, d_x, generate_dataset  # noqa: E402
from models.sbi_wrappers import run_sbi_model, train_sbi_model, sample_from_trained_sbi  # noqa: E402
from models.smmd import SMMD_Model, sliced_mmd_loss  # noqa: E402
from utilities import compute_swd_metric, run_gaussian_posterior_mcmc  # noqa: E402


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        return torch.device("cuda")
    return torch.device("cpu")


def _stereo_inverse(x_obs_3d: np.ndarray) -> np.ndarray:
    x_obs_3d = np.asarray(x_obs_3d, dtype=np.float32)
    if x_obs_3d.ndim != 2 or x_obs_3d.shape[1] != 3:
        raise ValueError(f"Expected x_obs_3d shape (n, 3), got {x_obs_3d.shape}")
    x_comp = x_obs_3d[:, 0]
    y_comp = x_obs_3d[:, 1]
    z_comp = x_obs_3d[:, 2]
    denom = 1.0 - z_comp
    denom[np.abs(denom) < 1e-6] = 1e-6
    x_2d = x_comp / denom
    y_2d = y_comp / denom
    return np.stack([x_2d, y_2d], axis=-1)


def _get_scheduler(optimizer, epochs: int):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=0)


def _resolve_batch_sizes(cfg: Dict, train_sizes: List[int], smoke: bool) -> Dict[int, int]:
    if "batch_sizes" in cfg:
        batch_sizes = [int(v) for v in cfg.get("batch_sizes", [])]
        if len(batch_sizes) != len(train_sizes):
            raise ValueError("config field 'batch_sizes' must have the same length as 'train_sizes'")
        mapping = {int(n_train): int(batch) for n_train, batch in zip(train_sizes, batch_sizes)}
    else:
        default_batch = int(cfg.get("batch_size", 256))
        mapping = {int(n_train): int(default_batch) for n_train in train_sizes}

    if smoke:
        mapping = {int(n_train): min(int(batch), 64) for n_train, batch in mapping.items()}
    return mapping


def train_smmd(
    model: SMMD_Model,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
    m_points: int,
    l_slices: int,
    l1_lambda: float,
    log_every_epochs: int = 10,
    *,
    n_obs: int,
) -> float:
    """Train the SMMD generator.

    Args:
        m_points: Number of posterior draws per observation (the ``M`` of the
            generator input ``z``).
        n_obs: Number of observations per dataset; sets the kernel bandwidth
            scale (``1 / n_obs``), matching the posterior contraction rate.
    """
    optimizer = optim.AdamW(model.parameters(), lr=float(lr))
    scheduler = _get_scheduler(optimizer, epochs)
    model.to(device)
    model.train()

    start_time = time.time()
    epochs = int(epochs)
    log_every_epochs = int(log_every_epochs) if log_every_epochs is not None else 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)

            optimizer.zero_grad()
            z = torch.randn(x_batch.size(0), int(m_points), model.d, device=device)
            theta_fake = model(x_batch, z)
            loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=int(l_slices), n_obs=n_obs)
            if l1_lambda and float(l1_lambda) > 0:
                l1_loss = 0.0
                for param in model.G.parameters():
                    l1_loss = l1_loss + torch.sum(torch.abs(param))
                loss = loss + float(l1_lambda) * l1_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        if log_every_epochs > 0:
            should_log = (epoch == 0) or ((epoch + 1) % log_every_epochs == 0) or (epoch + 1 == epochs)
            if should_log:
                avg_loss = epoch_loss / max(n_batches, 1)
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else float(lr)
                print(
                    f"[SMMD] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} lr={lr_now:.6g} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    return float(time.time() - start_time)


def _resolve_model_config(cfg: Dict, model_type: str) -> Dict:
    models_cfg = dict(cfg.get("models_config", {}))
    default_smmd_cfg = {
        "epochs": int(cfg.get("epochs", 200)),
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
        "summary_dim": int(cfg.get("summary_dim", 10)),
        "log_every_epochs": int(cfg.get("log_every_epochs", 10)),
        "smmd_M": int(cfg.get("smmd_M", 50)),
        "smmd_L": int(cfg.get("smmd_L", 20)),
        "smmd_l1_lambda": float(cfg.get("smmd_l1_lambda", 1e-4)),
    }
    model_cfg = dict(models_cfg.get(model_type, {}))
    if model_type == "smmd":
        return {**default_smmd_cfg, **model_cfg}
    if model_type in {"npe", "snpe_c", "snpe"}:
        alias_cfg = models_cfg.get("npe", None)
        if not model_cfg and alias_cfg is not None:
            model_cfg = dict(alias_cfg)
        return {
            "epochs": int(model_cfg.get("epochs", 200)),
            "sbi_rounds": int(model_cfg.get("sbi_rounds", 1)),
            "sims_per_round": int(model_cfg.get("sims_per_round", 1000)),
            "learning_rate": float(model_cfg.get("learning_rate", 5e-4)),
            "training_batch_size": int(model_cfg.get("training_batch_size", 200)),
            "validation_fraction": float(model_cfg.get("validation_fraction", 0.1)),
            "stop_after_epochs": int(model_cfg.get("stop_after_epochs", 20)),
            "show_train_summary": bool(model_cfg.get("show_train_summary", False)),
        }
    return model_cfg


def _model_display_name(model_type: str) -> str:
    model_type = str(model_type).lower()
    if model_type == "smmd":
        return "SMMD"
    if model_type in {"npe", "snpe_c", "snpe"}:
        return "NPE"
    return model_type.upper()


def main():
    # Config file selection: --config <path> > positional <path> (legacy) >
    # DATASIZE_CONFIG_PATH env var > config.json next to this script.
    _argv = sys.argv[1:]
    _cli_value = cli_config_arg(_argv)
    if _cli_value is None:
        _positional = [a for a in _argv if not a.startswith("-")]
        if _positional:
            _cli_value = _positional[0]
    cfg_path, _requested_cfg = resolve_config_path(
        os.path.dirname(os.path.abspath(__file__)),
        "DATASIZE_CONFIG_PATH",
        cli_value=_cli_value,
    )
    if cfg_path is None:
        raise FileNotFoundError(
            f"Config file not found (requested: {_requested_cfg or 'config.json'})."
        )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    results_dir = cfg.get("results_dir", "results_datasize_dependence")
    out_dir = os.path.join(os.path.dirname(__file__), str(results_dir))
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = bool(cfg.get("smoke", False)) or os.environ.get("DATASIZE_SMOKE", "").strip() == "1"
    n_obs_list: List[int] = [int(v) for v in cfg.get("n_obs_list", [25, 50, 100])]
    train_sizes: List[int] = [int(v) for v in cfg.get("train_sizes", [1000, 2000, 3000])]
    n_try = int(cfg.get("n_try", 1))
    batch_size_map = _resolve_batch_sizes(cfg, train_sizes, smoke=smoke)

    models_to_run: List[str] = [str(v).lower() for v in cfg.get("models_to_run", ["smmd", "npe"])]
    model_cfg_map = {model_name: _resolve_model_config(cfg, model_name) for model_name in models_to_run}

    n_samples_posterior = int(cfg.get("n_samples_posterior", 2000))
    metric_max_samples = int(cfg.get("metric_max_samples", 2000))
    swd_n_projections = int(cfg.get("swd_n_projections", 1000))
    swd_max_samples = int(cfg.get("swd_max_samples", metric_max_samples))
    true_post_cfg = dict(cfg.get("true_posterior_config", {}))

    device = _device()

    if smoke:
        n_obs_list = n_obs_list[:1]
        train_sizes = train_sizes[:1]
        batch_size_map = {int(train_sizes[0]): int(min(batch_size_map[int(train_sizes[0])], 64))}
        n_try = 1
        n_samples_posterior = min(n_samples_posterior, 128)
        metric_max_samples = min(metric_max_samples, 128)
        swd_n_projections = min(swd_n_projections, 64)
        swd_max_samples = min(swd_max_samples, 128)
        true_post_cfg = dict(true_post_cfg)
        true_post_cfg["n_draws"] = min(int(true_post_cfg.get("n_draws", 4)), 1)
        true_post_cfg["n_tune_chain"] = min(int(true_post_cfg.get("n_tune_chain", 800)), 50)
        true_post_cfg["chains"] = min(int(true_post_cfg.get("chains", 200)), 16)
        for model_name in models_to_run:
            mc = dict(model_cfg_map.get(model_name, {}))
            mc["epochs"] = min(int(mc.get("epochs", 2)), 2)
            if model_name == "smmd":
                mc["log_every_epochs"] = 1
            if model_name in {"npe", "snpe_c", "snpe"}:
                mc["sbi_rounds"] = min(int(mc.get("sbi_rounds", 1)), 1)
                mc["sims_per_round"] = min(int(mc.get("sims_per_round", 256)), 256)
                mc["training_batch_size"] = min(int(mc.get("training_batch_size", 64)), 64)
                mc["stop_after_epochs"] = min(int(mc.get("stop_after_epochs", 5)), 5)
            model_cfg_map[model_name] = mc

    raw_rows: List[Dict] = []
    agg_rows: List[Dict] = []

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "seed": seed,
        "n_obs_list": n_obs_list,
        "train_sizes": train_sizes,
        "batch_sizes": [int(batch_size_map[int(n_train)]) for n_train in train_sizes],
        "n_try": n_try,
        "models_to_run": models_to_run,
        "models_config": model_cfg_map,
        "n_samples_posterior": n_samples_posterior,
        "metric_max_samples": metric_max_samples,
        "swd_n_projections": swd_n_projections,
        "swd_max_samples": swd_max_samples,
        "true_posterior_config": true_post_cfg,
        "evaluation_metrics": ["swd_scaled_by_sqrt_n"],
        "evaluation_protocol": {
            "retrain_for_each_try": False,
            "regenerate_training_dataset_for_each_try": False,
            "sample_new_test_theta_for_each_try": False,
            "fixed_test_theta": [float(v) for v in TRUE_PARAMS],
            "sample_new_observation_for_each_try": True,
            "sample_true_posterior_for_each_try": True,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    raw_csv_path = os.path.join(out_dir, "results_raw.csv")
    agg_csv_path = os.path.join(out_dir, "results_agg.csv")

    for n_obs in n_obs_list:
        task = GaussianTask(n=int(n_obs))

        for n_train in train_sizes:
            batch_size = int(batch_size_map[int(n_train)])

            # --- Generate training dataset once per (n_obs, n_train) ---
            print(
                f"\n=== Generating training data: n_obs={int(n_obs)} | N_train={int(n_train)} ===",
                flush=True,
            )
            theta_train, x_train = generate_dataset(task=task, n_sims=int(n_train), n_obs=int(n_obs))
            x_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
            theta_tensor = torch.from_numpy(np.asarray(theta_train, dtype=np.float32))
            dataset = TensorDataset(x_tensor, theta_tensor)
            train_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)

            per_model = {
                model_name: {
                    "swd_vals": [],
                    "train_time": None,
                    "true_times": [],
                    "metric_times": [],
                    "eval_times": [],
                }
                for model_name in models_to_run
            }

            # --- Train models once per (n_obs, n_train) ---
            trained_models = {}
            for model_name in models_to_run:
                model_conf = dict(model_cfg_map.get(model_name, {}))
                model_label = _model_display_name(model_name)

                print(
                    f"\n=== Training {model_label} | n_obs={int(n_obs)} | N_train={int(n_train)} | batch={batch_size} ===",
                    flush=True,
                )

                if model_name == "smmd":
                    model = SMMD_Model(
                        summary_dim=int(model_conf.get("summary_dim", 10)),
                        d=d,
                        d_x=d_x,
                        n=int(n_obs),
                    )
                    print(
                        f"Training {model_label}: epochs={int(model_conf.get('epochs', 200))} "
                        f"batch_size={int(batch_size)} lr={float(model_conf.get('learning_rate', 1e-3))}",
                        flush=True,
                    )
                    t_train = train_smmd(
                        model=model,
                        train_loader=train_loader,
                        epochs=int(model_conf.get("epochs", 200)),
                        device=device,
                        lr=float(model_conf.get("learning_rate", 1e-3)),
                        m_points=int(model_conf.get("smmd_M", 50)),
                        l_slices=int(model_conf.get("smmd_L", 20)),
                        l1_lambda=float(model_conf.get("smmd_l1_lambda", 1e-4)),
                        log_every_epochs=int(model_conf.get("log_every_epochs", 10)),
                        n_obs=int(n_obs),
                    )
                    trained_models[model_name] = model
                    per_model[model_name]["train_time"] = float(t_train)

                elif model_name in {"npe", "snpe_c", "snpe"}:
                    effective_sims_per_round = int(theta_train.shape[0])
                    print(
                        f"Training {model_label}: rounds={int(model_conf.get('sbi_rounds', 1))} "
                        f"sims_per_round={effective_sims_per_round} "
                        f"epochs={int(model_conf.get('epochs', 200))}",
                        flush=True,
                    )
                    t_train0 = time.time()
                    trained = train_sbi_model(
                        model_type=model_name,
                        task=task,
                        device=str(device),
                        num_rounds=int(model_conf.get("sbi_rounds", 1)),
                        sims_per_round=effective_sims_per_round,
                        max_epochs=int(model_conf.get("epochs", 200)),
                        initial_training_data=(x_train, theta_train),
                        validation_fraction=float(model_conf.get("validation_fraction", 0.1)),
                        stop_after_epochs=int(model_conf.get("stop_after_epochs", 20)),
                        learning_rate=float(model_conf.get("learning_rate", 5e-4)),
                        training_batch_size=int(model_conf.get("training_batch_size", 200)),
                        show_train_summary=bool(model_conf.get("show_train_summary", False)),
                    )
                    t_train = float(time.time() - t_train0)
                    trained_models[model_name] = trained
                    per_model[model_name]["train_time"] = float(t_train)

                else:
                    raise ValueError(f"Unsupported model for DataSizeDependence: {model_name}")

            # --- Evaluate on n_try different test theta* observations ---
            for try_idx in range(n_try):
                print(
                    f"\n=== Evaluate n_obs={int(n_obs)} | N_train={int(n_train)} | try {try_idx+1}/{int(n_try)} ===",
                    flush=True,
                )

                # Fixed test theta* (same TRUE_PARAMS as the main Gaussian
                # experiment); only the observation is resampled per try.
                theta_true = TRUE_PARAMS
                x_obs_3d = task.simulator(theta_true[None, :], n_samples=int(n_obs))[0]
                x_obs_2d = _stereo_inverse(x_obs_3d)

                # Compute true posterior for this observation
                coeff = float(true_post_cfg.get("proposal_scale_coeff", 0.6))
                proposal_scale = float(coeff / np.sqrt(float(task.n)))
                print(
                    f"Sampling true posterior (RWM): chains={int(true_post_cfg.get('chains', 200))} "
                    f"tune={int(true_post_cfg.get('n_tune_chain', 800))} "
                    f"draws={int(true_post_cfg.get('n_draws', 4))} "
                    f"proposal_scale={proposal_scale:.6f}",
                    flush=True,
                )
                t_true0 = time.time()
                true_samples = run_gaussian_posterior_mcmc(
                    x_obs_2d,
                    task,
                    n_draws=int(true_post_cfg.get("n_draws", 4)),
                    n_tune_chain=int(true_post_cfg.get("n_tune_chain", 800)),
                    chains=int(true_post_cfg.get("chains", 200)),
                    proposal_scale=float(proposal_scale),
                )
                true_time_sec = float(time.time() - t_true0)
                for model_name in models_to_run:
                    model_conf = dict(model_cfg_map.get(model_name, {}))
                    model_label = _model_display_name(model_name)

                    # Sample from pre-trained model (no retraining)
                    if model_name == "smmd":
                        approx_samples = trained_models[model_name].sample_posterior(
                            np.asarray(x_obs_3d, dtype=np.float32), int(n_samples_posterior)
                        )
                    elif model_name in {"npe", "snpe_c", "snpe"}:
                        approx_samples = sample_from_trained_sbi(
                            trained_models[model_name],
                            np.asarray(x_obs_3d, dtype=np.float32),
                            int(n_samples_posterior),
                            mcmc_chains=int(model_conf.get("mcmc_chains", 100)),
                        )
                    else:
                        raise ValueError(f"Unsupported model for DataSizeDependence: {model_name}")

                    stats = per_model[model_name]
                    stats["true_times"].append(float(true_time_sec))

                    print(
                        f"Eval {model_label}: approximate posterior vs true posterior with sW ...",
                        flush=True,
                    )
                    metric_seed = (
                        int(seed)
                        + int(n_obs) * 100000
                        + int(n_train) * 100
                        + int(try_idx) * 10
                        + int(models_to_run.index(model_name))
                    )

                    rng = np.random.RandomState(int(metric_seed))
                    approx_samples_np = np.asarray(approx_samples, dtype=np.float32)
                    true_samples_np = np.asarray(true_samples, dtype=np.float32)
                    n_common = min(int(metric_max_samples), approx_samples_np.shape[0], true_samples_np.shape[0])

                    if n_common >= 2:
                        idx_approx = rng.choice(approx_samples_np.shape[0], size=n_common, replace=False)
                        idx_true = rng.choice(true_samples_np.shape[0], size=n_common, replace=False)
                        approx_sub = approx_samples_np[idx_approx]
                        true_sub = true_samples_np[idx_true]
                    else:
                        approx_sub = approx_samples_np
                        true_sub = true_samples_np

                    t_metric0 = time.time()
                    scale = float(np.sqrt(float(task.n)))
                    swd_val = float(
                        compute_swd_metric(
                            approx_sub * scale,
                            true_sub * scale,
                            n_projections=int(swd_n_projections),
                            max_samples=int(swd_max_samples),
                            seed=int(metric_seed),
                        )
                    )
                    metric_time_sec = float(time.time() - t_metric0)

                    stats["metric_times"].append(float(metric_time_sec))
                    stats["eval_times"].append(float(metric_time_sec + true_time_sec))
                    stats["swd_vals"].append(float(swd_val))

                    raw_rows.append(
                        {
                            "model_name": str(model_name),
                            "model_label": model_label,
                            "n_obs": int(n_obs),
                            "n_train": int(n_train),
                            "batch_size": int(batch_size),
                            "try": int(try_idx),
                            "theta_true_0": float(theta_true[0]),
                            "theta_true_1": float(theta_true[1]),
                            "theta_true_2": float(theta_true[2]),
                            "theta_true_3": float(theta_true[3]),
                            "theta_true_4": float(theta_true[4]),
                            "proposal_scale": float(proposal_scale),
                            "true_posterior_method": str(method),
                            "n_metric_samples": int(n_common),
                            "swd": float(swd_val),
                            "swd_scale": float(scale),
                            "train_time_sec": float(per_model[model_name]["train_time"]),
                            "true_posterior_time_sec": float(true_time_sec),
                            "metric_time_sec": float(metric_time_sec),
                            "eval_time_sec": float(metric_time_sec + true_time_sec),
                            "metric_seed": int(metric_seed),
                        }
                    )
                    pd.DataFrame(raw_rows).to_csv(raw_csv_path, index=False)
                    print(
                        f"Eval done ({model_label}): sW(scaled)={swd_val:.6f} "
                        f"true_post_time={true_time_sec:.2f}s metric_time={metric_time_sec:.2f}s",
                        flush=True,
                    )

            for model_name in models_to_run:
                stats = per_model[model_name]
                agg_rows.append(
                    {
                        "model_name": str(model_name),
                        "model_label": _model_display_name(model_name),
                        "n_obs": int(n_obs),
                        "n_train": int(n_train),
                        "batch_size": int(batch_size),
                        "n_try": int(n_try),
                        "swd_mean": float(np.mean(stats["swd_vals"])) if stats["swd_vals"] else float("nan"),
                        "swd_std": float(np.std(stats["swd_vals"], ddof=0)) if stats["swd_vals"] else float("nan"),
                        "train_time_mean_sec": float(stats["train_time"]) if stats["train_time"] is not None else float("nan"),
                        "train_time_std_sec": 0.0,
                        "true_posterior_time_mean_sec": float(np.mean(stats["true_times"])) if stats["true_times"] else float("nan"),
                        "metric_time_mean_sec": float(np.mean(stats["metric_times"])) if stats["metric_times"] else float("nan"),
                        "eval_time_mean_sec": float(np.mean(stats["eval_times"])) if stats["eval_times"] else float("nan"),
                    }
                )
                print(
                    f"Cell summary ({_model_display_name(model_name)}): "
                    f"| sW_mean={np.mean(stats['swd_vals']):.6f} sW_std={np.std(stats['swd_vals'], ddof=0):.6f}",
                    flush=True,
                )
            pd.DataFrame(agg_rows).to_csv(agg_csv_path, index=False)

    df_raw = pd.DataFrame(raw_rows)
    df_agg = pd.DataFrame(agg_rows)
    df_raw.to_csv(raw_csv_path, index=False)
    # Figure 2: aggregated sqrt(n)*sW per (training budget, observation count).
    df_agg.to_csv(agg_csv_path, index=False)

    # --- Font size configuration (paper-readable) ---
    FS_LABEL = 15
    FS_TICK = 12
    FS_TITLE = 14
    FS_LEGEND = 12

    model_order = [m for m in ["smmd", "npe"] if m in set(df_agg["model_name"].astype(str).tolist())]
    if not model_order:
        model_order = list(dict.fromkeys(df_agg["model_name"].astype(str).tolist()))
    # Colors matching the main Gaussian experiment
    color_map = {"smmd": "#DB1218", "npe": "#f19327", "snpe_c": "#f19327", "snpe": "#f19327"}

    ylabel_map = {"swd": r"$\mathrm{sW}$"}

    def _plot_metric_transposed(metric: str, out_name: str):
        """Transposed multi-panel: 1 row × len(train_sizes) columns, n on x-axis (linear)."""
        n_cols = len(train_sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, n_train in zip(axes, train_sizes):
            sub_n = df_agg[df_agg["n_train"] == int(n_train)]
            for model_name in model_order:
                sub = sub_n[sub_n["model_name"] == str(model_name)].sort_values("n_obs")
                if len(sub) == 0:
                    continue
                x_vals = sub["n_obs"].values.astype(np.float64)
                y_vals = sub[f"{metric}_mean"].values
                y_errs = sub[f"{metric}_std"].values
                c = color_map.get(str(model_name), None)
                label = _model_display_name(str(model_name))
                ax.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=6, color=c, label=label)
                ax.fill_between(
                    x_vals,
                    y_vals - y_errs,
                    y_vals + y_errs,
                    color=c,
                    alpha=0.2,
                )
            ax.set_xticks(n_obs_list)
            ax.set_xticklabels([str(int(v)) for v in n_obs_list], fontsize=FS_TICK)
            ax.set_xlabel(r"$n$", fontsize=FS_LABEL)
            ax.set_title(rf"$N_{{\mathrm{{train}}}} = {int(n_train)}$", fontsize=FS_TITLE)
            ax.tick_params(axis="y", labelsize=FS_TICK)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", framealpha=0.8, fontsize=FS_LEGEND)

        axes[0].set_ylabel(ylabel_map.get(metric, metric), fontsize=FS_LABEL)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved transposed figure: {os.path.join(out_dir, out_name)}", flush=True)

    # --- Generate figures ---
    # Transposed multi-panel: N_train as columns, n on x-axis (linear).
    # Figure 2: multi-panel sW vs n figure, one panel per training budget.
    _plot_metric_transposed(metric="swd", out_name="datasize_dependence_swd_transposed.png")


if __name__ == "__main__":
    main()
