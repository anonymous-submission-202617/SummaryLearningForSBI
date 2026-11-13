###############################################################################
# Radio (Turin channel) real-data application
#
# Infers 4 channel parameters from 801-bin impulse-response observations of
# a misspecified simulator, using SMMD / MMD / BayesFlow / DNN-ABC / NPE-RS
# with local refinement, over 20 replications on the observed data.
#
# Reproduces the predictive-performance figure and the RMSE metric reported
# at the end of the "Real Data Application" subsection.
#
# Outputs: results/ (see README.md).
###############################################################################

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
import argparse

# Make the experiment package and the repository root importable regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sbi_common.checkpoints import build_bayesflow_for_weights, released_weights, restore_torch
from sbi_common.config_utils import is_evaluation, resolve_config_path
from data_generation import load_task
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.sbi_wrappers import run_npe_rs
from utilities import refine_posterior_smmd, refine_posterior_bayesflow


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dataloader(x: np.ndarray, theta: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    theta_t = torch.from_numpy(np.asarray(theta, dtype=np.float32))
    ds = TensorDataset(x_t, theta_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_smmd_mmd(
    model,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    model_type: str,
    m: int,
    num_slices: int,
    n_time_steps: int,
    device,
    generator_l1: float = 0.0,
):
    _log(
        f"Training {model_type.upper()} | epochs={epochs}, lr={lr}, M={m}, "
        f"L={num_slices}, n_obs={n_time_steps}, device={device}"
    )
    model.to(device)
    if not hasattr(model, "theta_mean_buf") or not hasattr(model, "theta_std_buf"):
        raise ValueError("Model missing theta standardizer buffers.")
    if int(model.theta_mean_buf.numel()) != int(model.d) or int(model.theta_std_buf.numel()) != int(model.d):
        raise ValueError("theta standardizer buffers are not set.")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_batches = len(train_loader)
    t0 = time.perf_counter()
    for ep in range(int(epochs)):
        ep_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            opt.zero_grad()
            with torch.enable_grad():
                theta_batch = (theta_batch - model.theta_mean_buf) / model.theta_std_buf
                z = torch.randn(x_batch.size(0), m, model.d, device=device)
                theta_fake = model(x_batch, z)
                if model_type == "smmd":
                    loss_main = sliced_mmd_loss(theta_batch, theta_fake, num_slices=num_slices, n_time_steps=n_time_steps)
                else:
                    raise ValueError(f"Unsupported model_type: {model_type}")
                if generator_l1 > 0.0:
                    l1 = torch.tensor(0.0, device=device)
                    for p in model.G.parameters():
                        if p is not None:
                            l1 = l1 + p.abs().sum()
                    loss = loss_main + float(generator_l1) * l1
                else:
                    loss = loss_main
                loss.backward()
                opt.step()
            ep_loss += float(loss.detach().cpu().item())
        if ep == 0 or (ep + 1) % 10 == 0 or ep == int(epochs) - 1:
            elapsed = time.perf_counter() - t0
            _log(
                f"{model_type.upper()} epoch {ep + 1}/{epochs} | avg_loss={ep_loss / max(n_batches, 1):.6f} "
                f"| elapsed={elapsed:.1f}s"
            )
    return model


def train_bayesflow(
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    d: int,
    d_x: int,
    summary_dim: int,
    *,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
):
    os.environ["KERAS_BACKEND"] = "torch"
    from models import bayesflow_net as bfn
    keras = bfn.keras
    build_bayesflow_model = bfn.build_bayesflow_model

    _log(f"Training BAYESFLOW | epochs={epochs}, lr={lr}, summary_dim={summary_dim}")
    model = build_bayesflow_model(d, d_x, summary_dim=summary_dim, theta_mean=theta_mean, theta_std=theta_std)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, int(epochs) * steps_per_epoch)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=float(lr),
        decay_steps=total_steps,
        alpha=0.0,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    model.compile(optimizer=optimizer)
    n_batches = len(train_loader)
    theta_mean_t = torch.as_tensor(np.asarray(theta_mean, dtype=np.float32))
    theta_std_t = torch.as_tensor(np.asarray(theta_std, dtype=np.float32))
    t0 = time.perf_counter()
    for ep in range(int(epochs)):
        ep_loss = 0.0
        for x_batch, theta_batch in train_loader:
            theta_batch = theta_batch.float()
            theta_z = (theta_batch - theta_mean_t) / theta_std_t
            batch_dict = {
                "inference_variables": theta_z,
                "summary_variables": x_batch.float(),
            }
            out = model.train_step(batch_dict)
            if isinstance(out, dict) and "loss" in out:
                ep_loss += float(out["loss"])
        if ep == 0 or (ep + 1) % 10 == 0 or ep == int(epochs) - 1:
            elapsed = time.perf_counter() - t0
            _log(
                f"BAYESFLOW epoch {ep + 1}/{epochs} | avg_loss={ep_loss / max(n_batches, 1):.6f} "
                f"| elapsed={elapsed:.1f}s"
            )
    return model


def sample_theta_smmd(model, x_obs: np.ndarray, n_samples: int, task=None) -> np.ndarray:
    x_obs = np.asarray(x_obs, dtype=np.float32)
    n_samples = int(n_samples)
    if task is None:
        return np.asarray(model.sample_posterior(x_obs, n_samples), dtype=np.float32)

    out = []
    remaining = n_samples
    max_rounds = 50
    oversample = 3
    for _ in range(max_rounds):
        if remaining <= 0:
            break
        draw = int(max(remaining, 1) * oversample)
        cand = np.asarray(model.sample_posterior(x_obs, draw), dtype=np.float32)
        cand = cand[np.isfinite(task.log_prior(cand))]
        if cand.shape[0] == 0:
            continue
        take = min(remaining, cand.shape[0])
        out.append(cand[:take])
        remaining -= take
    if not out:
        return np.asarray(model.sample_posterior(x_obs, n_samples), dtype=np.float32)
    return np.concatenate(out, axis=0)


def sample_theta_bayesflow(model, x_obs: np.ndarray, n_samples: int, task=None) -> np.ndarray:
    x_obs_cpu = np.asarray(x_obs, dtype=np.float32)
    if x_obs_cpu.ndim == 1:
        x_obs_cpu = x_obs_cpu[None, None, :]
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[None, :, :]
    n_samples = int(n_samples)

    def _draw(k: int) -> np.ndarray:
        x_obs_rep = np.repeat(x_obs_cpu, repeats=int(k), axis=0)
        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
        if isinstance(post, dict):
            post = post.get("inference_variables", post)
        if hasattr(post, "numpy"):
            post = post.numpy()
        return np.asarray(post, dtype=np.float32).reshape(int(k), -1)

    if task is None:
        return _draw(n_samples)

    out = []
    remaining = n_samples
    max_rounds = 50
    oversample = 3
    for _ in range(max_rounds):
        if remaining <= 0:
            break
        draw = int(max(remaining, 1) * oversample)
        cand = _draw(draw)
        cand = cand[np.isfinite(task.log_prior(cand))]
        if cand.shape[0] == 0:
            continue
        take = min(remaining, cand.shape[0])
        out.append(cand[:take])
        remaining -= take
    if not out:
        return _draw(n_samples)
    return np.concatenate(out, axis=0)


def filter_valid(task, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    logp = task.log_prior(theta)
    mask = np.isfinite(logp)
    if not np.any(mask):
        return theta
    return theta[mask]


def plot_ppc(
    sims: np.ndarray,
    obs_cases: np.ndarray,
    out_path: Path,
    logscale: bool = True,
    color: str = "#132a70",
    figsize: tuple[int, int] = (12, 6),
    font_size: int = 18,
):
    obs_cases = np.asarray(obs_cases, dtype=np.float32).reshape(-1)
    sims = np.asarray(sims, dtype=np.float32)
    qs_50 = np.quantile(sims, q=[0.25, 0.75], axis=0)
    qs_90 = np.quantile(sims, q=[0.05, 0.95], axis=0)
    qs_95 = np.quantile(sims, q=[0.025, 0.975], axis=0)
    med = np.median(sims, axis=0)

    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(med, label="Median predicted", color=color)
    ax.plot(obs_cases, marker="o", label="Reported (obs)", color="black", linestyle="dashed", alpha=0.8)
    ax.fill_between(range(len(obs_cases)), qs_50[0], qs_50[1], color=color, alpha=0.5, label="50% CI")
    ax.fill_between(range(len(obs_cases)), qs_90[0], qs_90[1], color=color, alpha=0.3, label="90% CI")
    ax.fill_between(range(len(obs_cases)), qs_95[0], qs_95[1], color=color, alpha=0.1, label="95% CI")
    ax.grid(color="grey", linestyle="-", linewidth=0.25, alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Frequency index (IFFT domain)")
    ax.set_ylabel("Power (dB)")
    ax.minorticks_off()
    if logscale:
        if np.any(med <= 0.0) or np.any(obs_cases <= 0.0):
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_yscale("log")
    ax.legend(fontsize=font_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=350)
    plt.close(fig)
    _log(f"Saved PPC plot: {out_path}")


def plot_predictive_mean_std(series: list[dict], obs: np.ndarray, out_path: Path):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    xp = np.linspace(0.0, 200.0, obs.shape[0], dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.plot(xp, obs, color="black", linewidth=3, label="Real data")
    for s in series:
        mean = np.asarray(s["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(s["std"], dtype=np.float32).reshape(-1)
        label = str(s["label"])
        color = s.get("color")
        linestyle = s.get("linestyle", "-")
        linewidth = int(s.get("linewidth", 3))
        alpha = float(s.get("alpha", 0.3))
        ax.plot(xp, mean, color=color, linewidth=linewidth, linestyle=linestyle, alpha=1.0, label=label)
        ax.fill_between(xp, mean - std, mean + std, color=color, alpha=alpha)
    ax.set_xlabel("Time Delay [ns]")
    ax.set_ylabel("Power [db]")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    _log(f"Saved predictive mean±std plot: {out_path}")


def plot_summary_pairwise(stats_sim: np.ndarray, stats_obs: np.ndarray, out_path: Path, max_dim: int = 8):
    stats_sim = np.asarray(stats_sim, dtype=np.float32)
    stats_obs = np.asarray(stats_obs, dtype=np.float32).reshape(-1)
    d_full = int(stats_sim.shape[1])
    d = int(min(d_full, int(max_dim)))
    stats_sim = stats_sim[:, :d]
    stats_obs = stats_obs[:d]
    fig, axes = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d), sharex="col", sharey="row")
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(stats_sim[:, j], bins=30, color="#1f77b4", alpha=0.6, density=True)
                ax.axvline(stats_obs[j], color="orange", linewidth=2.5)
            else:
                ax.scatter(stats_sim[:, j], stats_sim[:, i], s=4, alpha=0.25, color="#1f77b4", edgecolors="none")
                ax.scatter([stats_obs[j]], [stats_obs[i]], s=40, color="orange", edgecolors="black", linewidths=0.5)
            if i == d - 1:
                ax.set_xlabel(f"s{j}")
            if j == 0:
                ax.set_ylabel(f"s{i}")
            ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=250)
    plt.close(fig)
    _log(f"Saved summary pairwise plot: {out_path}")


def predictive_rmse_mean_curve(predictive_data_db: np.ndarray, obs_mean_db: np.ndarray) -> float:
    pred_mean = np.mean(np.asarray(predictive_data_db, dtype=np.float32), axis=0)
    obs_mean = np.asarray(obs_mean_db, dtype=np.float32).reshape(-1)
    if pred_mean.shape != obs_mean.shape:
        raise ValueError(f"RMSE shape mismatch: pred_mean {pred_mean.shape} vs obs_mean {obs_mean.shape}")
    return float(np.sqrt(np.mean((pred_mean - obs_mean) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    # Config file selection: --config <path> > RADIO_CONFIG_PATH env var >
    # config.json next to this script.
    cfg_path, _requested_cfg = resolve_config_path(
        str(project_root), "RADIO_CONFIG_PATH", cli_value=args.config
    )
    if cfg_path is None:
        raise FileNotFoundError(
            f"Config file not found (requested: {_requested_cfg or 'config.json'})."
        )
    cfg = load_config(cfg_path)
    evaluation = is_evaluation(cfg)
    weights_dir = released_weights(project_root, "radio")
    task, x_obs_mat = load_task(
        str(project_root), config_path=str(cfg_path),
        observation_path=weights_dir / "data" / "turin_obs.npy" if evaluation else None,
    )
    if evaluation and "bayesflow" in cfg.get("models_to_run", ["bayesflow"]):
        with (project_root / "bayesflow_training_seeds.json").open() as f:
            bayesflow_training_seeds = json.load(f)
    x_obs_mat = np.asarray(x_obs_mat, dtype=np.float32)[: int(task.N_rx), : int(task.Ns)]
    x_obs_mean = np.mean(x_obs_mat, axis=0).astype(np.float32)
    n_time_steps = int(task.Ns)
    n_realizations = int(task.N_rx)
    _log(
        f"Loaded observed Turin data | Ns={n_time_steps} | N_rx={n_realizations} | x_obs shape={tuple(x_obs_mat.shape)}"
    )

    n_rounds = int(cfg.get("n_rounds", 1))
    base_seed = int(cfg.get("seed", 0))
    # Results are written next to the script (cwd-independent), or to the
    # directory requested via RADIO_RESULTS_DIR (used e.g. for smoke tests).
    results_dir = "results_evaluation" if evaluation else "results"
    results_root = Path(os.environ.get("RADIO_RESULTS_DIR", str(project_root / results_dir)))
    results_root.mkdir(parents=True, exist_ok=True)
    rmse_rows_all: list[dict] = []
    for r in range(int(n_rounds)):
        round_seed = base_seed + r
        np.random.seed(round_seed)
        torch.manual_seed(round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(round_seed)

        result_base = results_root / f"round_{r + 1:02d}"
        result_base.mkdir(parents=True, exist_ok=True)
        with open(result_base / "config_used.json", "w", encoding="utf-8") as f:
            cfg_used = dict(cfg)
            cfg_used["seed"] = round_seed
            cfg_used["round"] = r + 1
            json.dump(cfg_used, f, indent=2)

        _log(f"========== Round {r + 1}/{n_rounds} | seed={round_seed} ==========")

        dataset_size = int(cfg.get("dataset_size", 25600))
        batch_size = int(cfg.get("batch_size", 128))
        lr = float(cfg.get("learning_rate", 1e-3))
        n_samples_posterior = int(cfg.get("n_samples_posterior", 1000))

        if not evaluation:
            _log(f"Generating training data | dataset_size={dataset_size} ...")
            theta_train = task.sample_prior(dataset_size)
            # Simulate in chunks with progress logging (simulator is the bottleneck)
            sim_chunk_size = 1000
            x_chunks = []
            for ch_start in range(0, dataset_size, sim_chunk_size):
                ch_end = min(dataset_size, ch_start + sim_chunk_size)
                x_chunk = task.simulator(theta_train[ch_start:ch_end])
                x_chunks.append(x_chunk)
                _log(f"  Simulated {ch_end}/{dataset_size} samples")
            x_train = np.concatenate(x_chunks, axis=0)
            _log(f"Generated dataset: theta {theta_train.shape} | x {x_train.shape}")

            train_loader = make_dataloader(x_train, theta_train, batch_size=batch_size, shuffle=True)
            theta_mean = np.mean(theta_train, axis=0).astype(np.float32)
            theta_std = np.std(theta_train, axis=0).astype(np.float32)
            theta_std = np.maximum(theta_std, 1e-12).astype(np.float32)

        models_to_run = list(cfg.get("models_to_run", ["smmd", "bayesflow", "dnnabc", "npe_rs"]))
        allowed = {"smmd", "bayesflow", "dnnabc", "npe_rs"}
        models_to_run = [m for m in models_to_run if m in allowed]
        models_config = cfg.get("models_config", {})
        smmd_mmd_cfg = cfg.get("smmd_mmd_config", {})
        refinement_cfg = cfg.get("refinement_config", {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(f"Experiment config | models_to_run={models_to_run} | device={device}")

        m = int(smmd_mmd_cfg.get("M", 50))
        num_slices = int(smmd_mmd_cfg.get("L", 20))
        compare_series = []
        compare_series_refined = {}
        rmse_rows: list[dict] = []

        def record_eval_metrics(label: str, predictive_data_db: np.ndarray) -> float:
            rmse_mean_curve_val = predictive_rmse_mean_curve(predictive_data_db, x_obs_mean)
            rmse_rows.append(
                {
                    "round": int(r + 1),
                    "model": str(label),
                    "rmse_mean_curve": float(rmse_mean_curve_val),
                    "n_pred": int(np.asarray(predictive_data_db).shape[0]),
                }
            )
            return float(rmse_mean_curve_val)

        color_map = {
            "smmd": "C0",
            "bayesflow": "C1",
            "dnnabc": "C2",
            "npe_rs": "C3",
        }

        for model_name in models_to_run:
            checkpoint = weights_dir / model_name / f"{model_name}_radio_seed{round_seed:02d}"
            _log(f"========== Running model: {model_name.upper()} ==========")

            if model_name == "smmd":
                mc = models_config.get("smmd", {})
                model = SMMD_Model(summary_dim=int(mc.get("summary_dim", 16)), d=task.d, d_x=1, n=n_realizations)
                if evaluation:
                    model.set_theta_standardizer(np.zeros(task.d), np.ones(task.d))
                    restore_torch(model, checkpoint.with_suffix(".pt"), device)
                else:
                    model.set_theta_standardizer(theta_mean, theta_std)
                    model = train_smmd_mmd(
                        model,
                        train_loader,
                        epochs=int(mc.get("epochs", 200)),
                        lr=lr,
                        model_type="smmd",
                        m=m,
                        num_slices=num_slices,
                        n_time_steps=n_time_steps,
                        device=device,
                        generator_l1=float(cfg.get("generator_l1", 1e-6)),
                    )

                    with torch.no_grad():
                        stats_fn = model.compute_stats if hasattr(model, "compute_stats") else model.T
                        n_pair = int(min(2000, x_train.shape[0]))
                        idx = np.random.choice(x_train.shape[0], size=n_pair, replace=False)
                        x_batch = torch.from_numpy(x_train[idx]).float().to(device)
                        stats_sim = stats_fn(x_batch).detach().cpu().numpy()
                        x_obs_t = torch.from_numpy(x_obs_mat[None, :, :]).float().to(device)
                        stats_obs = stats_fn(x_obs_t).detach().cpu().numpy().reshape(-1)
                    plot_summary_pairwise(stats_sim, stats_obs, result_base / "smmd_summary_pairwise.png", max_dim=8)

                theta_post = filter_valid(task, sample_theta_smmd(model, x_obs_mat, n_samples_posterior, task=task))
                sims = task.simulator(theta_post[:n_samples_posterior], n_rx=1).mean(axis=1)
                rmse_smmd_mean_curve = record_eval_metrics("SMMD", sims)
                _log(f"Evaluation | SMMD | RMSE(mean_curve) = {rmse_smmd_mean_curve:.6f}")
                compare_series.append(
                    {
                        "label": "SMMD",
                        "mean": np.mean(sims, axis=0),
                        "std": np.std(sims, axis=0),
                        "color": color_map["smmd"],
                        "linestyle": "-",
                    }
                )

                refined = refine_posterior_smmd(
                    model=model,
                    x_obs=x_obs_mat,
                    task=task,
                    n_chains=int(refinement_cfg.get("n_chains", 1000)),
                    n_samples=int(refinement_cfg.get("n_samples_per_chain", 1)),
                    burn_in=int(refinement_cfg.get("burn_in", 49)),
                    thin=int(refinement_cfg.get("thin", 1)),
                    epsilon=None,
                    proposal_std=float(refinement_cfg.get("proposal_std", 0.2)),
                    device=device,
                    epsilon_quantile=float(refinement_cfg.get("epsilon_quantile", 0.02)),
                    bandwidth_n_samples=int(smmd_mmd_cfg.get("bandwidth_n_samples", 2000)),
                )
                refined = filter_valid(task, refined)
                sims_ref = None
                if refined.shape[0] > 0:
                    sims_ref = task.simulator(refined[:n_samples_posterior], n_rx=1).mean(axis=1)
                    rmse_smmd_ref_mean_curve = record_eval_metrics("SMMD(Refined)", sims_ref)
                    _log(f"Evaluation | SMMD(Refined) | RMSE(mean_curve) = {rmse_smmd_ref_mean_curve:.6f}")
                    compare_series_refined["SMMD"] = {
                        "label": "SMMD(Refined)",
                        "mean": np.mean(sims_ref, axis=0),
                        "std": np.std(sims_ref, axis=0),
                        "color": color_map["smmd"],
                        "linestyle": "-",
                    }

                series = [
                    {"label": "SMMD", "mean": np.mean(sims, axis=0), "std": np.std(sims, axis=0), "color": "C1"},
                ]
                if sims_ref is not None:
                    series.append(
                        {
                            "label": "SMMD(Refined)",
                            "mean": np.mean(sims_ref, axis=0),
                            "std": np.std(sims_ref, axis=0),
                            "color": "C0",
                        }
                    )
                plot_predictive_mean_std(series, x_obs_mean, result_base / "smmd_ppc.png")

            elif model_name == "bayesflow":
                mc = models_config.get("bayesflow", {})
                if evaluation:
                    from models.bayesflow_net import build_bayesflow_model

                    # The archive stores network weights only. Recreate the
                    # parameter scaling using the original training prior draws.
                    rng_state = np.random.get_state()
                    try:
                        np.random.seed(bayesflow_training_seeds[str(round_seed)])
                        theta = task.sample_prior(19200)
                    finally:
                        np.random.set_state(rng_state)
                    model = build_bayesflow_model(
                        task.d, 1, summary_dim=int(mc.get("summary_dim", 16)),
                        theta_mean=theta.mean(axis=0), theta_std=np.maximum(theta.std(axis=0), 1e-12),
                        standardize=None,
                    )
                    build_bayesflow_for_weights(model, x_obs_mat, task.d)
                    with np.load(checkpoint.with_name(checkpoint.name + "_weights.npz")) as archive:
                        model.set_weights([archive[f"arr_{i}"] for i in range(len(archive.files))])
                else:
                    model = train_bayesflow(
                        train_loader,
                        epochs=int(mc.get("epochs", 200)),
                        lr=lr,
                        d=task.d,
                        d_x=1,
                        summary_dim=int(mc.get("summary_dim", 16)),
                        theta_mean=theta_mean,
                        theta_std=theta_std,
                    )

                    def _bf_to_numpy(v):
                        if isinstance(v, np.ndarray):
                            return v
                        if hasattr(v, "detach"):
                            return v.detach().cpu().numpy()
                        if hasattr(v, "numpy"):
                            return v.numpy()
                        return np.asarray(v)

                    try:
                        n_pair = int(min(2000, x_train.shape[0]))
                        idx = np.random.choice(x_train.shape[0], size=n_pair, replace=False)
                        stats_sim = _bf_to_numpy(model.summary_network(np.asarray(x_train[idx], dtype=np.float32)))
                        obs_in = np.asarray(x_obs_mat[None, :, :], dtype=np.float32)
                        stats_obs = _bf_to_numpy(model.summary_network(obs_in))
                        plot_summary_pairwise(
                            np.asarray(stats_sim, dtype=np.float32),
                            np.asarray(stats_obs, dtype=np.float32).reshape(-1),
                            result_base / "bayesflow_summary_pairwise.png",
                            max_dim=8,
                        )
                    except Exception as e:
                        _log(f"Skipped BayesFlow summary pairwise plot due to error: {e}")

                theta_post = filter_valid(task, sample_theta_bayesflow(model, x_obs_mat, n_samples_posterior, task=task))
                sims = task.simulator(theta_post[:n_samples_posterior], n_rx=1).mean(axis=1)
                rmse_bf_mean_curve = record_eval_metrics("BayesFlow", sims)
                _log(f"Evaluation | BayesFlow | RMSE(mean_curve) = {rmse_bf_mean_curve:.6f}")
                compare_series.append(
                    {
                        "label": "BayesFlow",
                        "mean": np.mean(sims, axis=0),
                        "std": np.std(sims, axis=0),
                        "color": color_map["bayesflow"],
                        "linestyle": "-",
                    }
                )

                refined = refine_posterior_bayesflow(
                    model=model,
                    x_obs=x_obs_mat,
                    task=task,
                    n_chains=int(refinement_cfg.get("n_chains", 1000)),
                    n_samples=int(refinement_cfg.get("n_samples_per_chain", 1)),
                    burn_in=int(refinement_cfg.get("burn_in", 49)),
                    thin=int(refinement_cfg.get("thin", 1)),
                    epsilon=None,
                    proposal_std=float(refinement_cfg.get("proposal_std", 0.2)),
                    epsilon_quantile=float(refinement_cfg.get("epsilon_quantile", 0.02)),
                    bandwidth_n_samples=int(smmd_mmd_cfg.get("bandwidth_n_samples", 2000)),
                    device=torch.device("cpu"),
                )
                refined = filter_valid(task, refined)
                sims_ref = None
                if refined.shape[0] > 0:
                    sims_ref = task.simulator(refined[:n_samples_posterior], n_rx=1).mean(axis=1)
                    rmse_bf_ref_mean_curve = record_eval_metrics("BayesFlow(Refined)", sims_ref)
                    _log(f"Evaluation | BayesFlow(Refined) | RMSE(mean_curve) = {rmse_bf_ref_mean_curve:.6f}")
                    compare_series_refined["BayesFlow"] = {
                        "label": "BayesFlow(Refined)",
                        "mean": np.mean(sims_ref, axis=0),
                        "std": np.std(sims_ref, axis=0),
                        "color": color_map["bayesflow"],
                        "linestyle": "-",
                    }

                series = [
                    {"label": "BayesFlow", "mean": np.mean(sims, axis=0), "std": np.std(sims, axis=0), "color": "C1"},
                ]
                if sims_ref is not None:
                    series.append(
                        {
                            "label": "BayesFlow(Refined)",
                            "mean": np.mean(sims_ref, axis=0),
                            "std": np.std(sims_ref, axis=0),
                            "color": "C0",
                        }
                    )
                plot_predictive_mean_std(series, x_obs_mean, result_base / "bayesflow_ppc.png")

            elif model_name == "dnnabc":
                mc = models_config.get("dnnabc", {})
                model = DNNABC_Model(d=task.d, d_x=1)
                if evaluation:
                    restore_torch(model, checkpoint.with_suffix(".pt"), device)
                else:
                    train_dnnabc(model, train_loader, epochs=int(mc.get("epochs", 200)), device=device)
                theta_post = abc_rejection_sampling(
                    model,
                    x_obs_mat,
                    task,
                    n_samples=n_samples_posterior,
                    n_pool=int(mc.get("n_pool", 50000)),
                    device=device,
                )
                theta_post = filter_valid(task, theta_post)
                sims = task.simulator(theta_post[:n_samples_posterior], n_rx=1).mean(axis=1)
                rmse_abc_mean_curve = record_eval_metrics("DNN-ABC", sims)
                _log(f"Evaluation | DNN-ABC | RMSE(mean_curve) = {rmse_abc_mean_curve:.6f}")
                plot_predictive_mean_std(
                    [
                        {
                            "label": "DNN-ABC",
                            "mean": np.mean(sims, axis=0),
                            "std": np.std(sims, axis=0),
                            "color": color_map["dnnabc"],
                        }
                    ],
                    x_obs_mean,
                    result_base / "dnnabc_ppc.png",
                )
                compare_series.append(
                    {
                        "label": "DNN-ABC",
                        "mean": np.mean(sims, axis=0),
                        "std": np.std(sims, axis=0),
                        "color": color_map["dnnabc"],
                        "linestyle": "-",
                    }
                )

            elif model_name == "npe_rs":
                _log("Running NPE-RS (new SBI SNPE+MAF with robustness regularizer)...")
                theta_post = run_npe_rs(
                    task=task,
                    x_obs=x_obs_mat,
                    cfg=cfg,
                    theta_train_np=None if evaluation else theta_train,
                    x_train_np=None if evaluation else x_train,
                    checkpoint_path=checkpoint.with_name(checkpoint.name + "_posterior_estimator.pt") if evaluation else None,
                )
                theta_post = filter_valid(task, theta_post)
                sims = task.simulator(theta_post[:n_samples_posterior], n_rx=1).mean(axis=1)
                rmse_npe_mean_curve = record_eval_metrics("NPE-RS", sims)
                _log(f"Evaluation | NPE-RS | RMSE(mean_curve) = {rmse_npe_mean_curve:.6f}")
                plot_predictive_mean_std(
                    [
                        {
                            "label": "NPE-RS",
                            "mean": np.mean(sims, axis=0),
                            "std": np.std(sims, axis=0),
                            "color": color_map["npe_rs"],
                        }
                    ],
                    x_obs_mean,
                    result_base / "npe_rs_ppc.png",
                )
                compare_series.append(
                    {
                        "label": "NPE-RS",
                        "mean": np.mean(sims, axis=0),
                        "std": np.std(sims, axis=0),
                        "color": color_map["npe_rs"],
                        "linestyle": "-",
                    }
                )

            else:
                raise ValueError(f"Unknown model: {model_name}")
            _log(f"Finished model: {model_name.upper()}")

        if compare_series:
            _log("Compare plot series: " + ", ".join([str(s.get("label")) for s in compare_series]))
            # Predictive-performance figure: posterior-predictive mean +/- std vs observed series.
            plot_predictive_mean_std(compare_series, x_obs_mean, result_base / "compare_predictive_mean_std.png")
            if compare_series_refined:
                by_label = {str(s.get("label")): s for s in compare_series}
                refined_plot = []
                refined_plot.append(compare_series_refined.get("SMMD", by_label.get("SMMD")))
                refined_plot.append(compare_series_refined.get("BayesFlow", by_label.get("BayesFlow")))
                refined_plot.append(by_label.get("DNN-ABC"))
                refined_plot.append(by_label.get("NPE-RS"))
                refined_plot = [s for s in refined_plot if s is not None]
                plot_predictive_mean_std(
                    refined_plot, x_obs_mean, result_base / "compare_predictive_mean_std_refined.png"
                )

        if rmse_rows:
            rmse_rows_all.extend(rmse_rows)

    if rmse_rows_all:
        # RMSE metric reported in the manuscript: per-round predictive RMSE,
        # averaged over the 20 rounds.
        out_all_rounds = results_root / "rmse_all_rounds.csv"
        with open(out_all_rounds, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["round", "model", "rmse_mean_curve", "n_pred"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rmse_rows_all:
                w.writerow({k: row.get(k) for k in fieldnames})
        _log(f"Saved all-round RMSE table: {out_all_rounds}")

        summary_rows = []
        for model_name in sorted({str(row["model"]) for row in rmse_rows_all}):
            vals = np.asarray(
                [float(row["rmse_mean_curve"]) for row in rmse_rows_all if str(row["model"]) == model_name],
                dtype=np.float64,
            )
            if vals.size == 0:
                continue
            summary_rows.append(
                {
                    "model": model_name,
                    "n_rounds": int(vals.size),
                    "rmse_mean_curve_mean": float(np.mean(vals)),
                    "rmse_mean_curve_median": float(np.median(vals)),
                    "rmse_mean_curve_std": float(np.std(vals, ddof=0)),
                }
            )

        out_summary = results_root / "rmse_summary_stats.csv"
        with open(out_summary, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "model",
                "n_rounds",
                "rmse_mean_curve_mean",
                "rmse_mean_curve_median",
                "rmse_mean_curve_std",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        _log(f"Saved RMSE summary stats: {out_summary}")

    _log("All done.")


if __name__ == "__main__":
    main()
