import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data_generation import RadioTask


def _load_obs(obs_path: Path, ns: int, n_rx: int) -> np.ndarray:
    obs = np.load(obs_path).astype(np.float32)
    if obs.ndim == 3:
        obs = obs.reshape(obs.shape[-2], obs.shape[-1])
    if obs.ndim == 1 and obs.size % ns == 0:
        obs = obs.reshape(-1, ns)
    if obs.ndim != 2:
        obs = obs.reshape(-1)
        obs = obs[: int(n_rx) * int(ns)].reshape(int(n_rx), int(ns))
    obs = obs[: int(n_rx), : int(ns)]
    return obs.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=801)
    parser.add_argument("--n-rx-obs", type=int, default=100)
    parser.add_argument("--n-rx-sim", type=int, default=1)
    parser.add_argument("--b", type=float, default=4e9)
    parser.add_argument("--tau0", type=float, default=6e-9)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--lower", type=float, nargs=4, default=None)
    parser.add_argument("--upper", type=float, nargs=4, default=None)
    parser.add_argument("--obs-path", type=str, default="data/turin_obs.npy")
    parser.add_argument("--out", type=str, default="figures/prior_samples_vs_obs.png")
    args = parser.parse_args()

    task = RadioTask(Ns=int(args.ns), N_rx=int(args.n_rx_obs), B=float(args.b), tau0=float(args.tau0))
    lower = np.asarray(args.lower, dtype=np.float32).reshape(4) if args.lower is not None else task.lower
    upper = np.asarray(args.upper, dtype=np.float32).reshape(4) if args.upper is not None else task.upper
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(upper > lower):
        raise ValueError(f"Invalid bounds. lower={lower}, upper={upper}")
    obs = _load_obs(Path(args.obs_path), ns=int(task.Ns), n_rx=int(task.N_rx))
    obs_mean = obs.mean(axis=0)

    if args.lower is None and args.upper is None:
        theta = task.sample_prior(int(args.n_samples))
    else:
        log_lower = np.log(lower.astype(np.float32))
        log_upper = np.log(upper.astype(np.float32))
        u = np.random.rand(int(args.n_samples), 4).astype(np.float32)
        theta = np.exp(log_lower + u * (log_upper - log_lower)).astype(np.float32)

    sims = task.simulator(theta, n_rx=int(args.n_rx_sim))
    sims_mean = sims.mean(axis=1)

    xp = np.linspace(0.0, 200.0, int(task.Ns), dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    for i in range(sims_mean.shape[0]):
        ax.plot(xp, sims_mean[i], color="#9ecae1", linewidth=1, alpha=0.25, zorder=1)
    ax.plot(xp, obs_mean, color="black", linewidth=3, label="Real data", zorder=3)
    ax.set_xlabel("Time Delay [ns]")
    ax.set_ylabel("Power [db]")
    ax.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
