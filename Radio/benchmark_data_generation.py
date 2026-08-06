import argparse
import time
from pathlib import Path

import numpy as np

from data_generation import load_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--plot_n", type=int, default=30)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    cfg_path = args.config or str(project_root / "config.json")

    task, x_obs = load_task(str(project_root), config_path=cfg_path)

    n = int(args.n) if args.n is not None else int(1024)
    ns = int(task.Ns)

    t0 = time.perf_counter()
    theta = task.sample_prior(n)
    t1 = time.perf_counter()
    x = task.simulator(theta)
    t2 = time.perf_counter()

    theta_s = t1 - t0
    sim_s = t2 - t1
    total_s = t2 - t0

    print("Ns:", ns)
    print("N_rx:", int(task.N_rx))
    print("B:", float(task.B))
    print("simulator_backend:", getattr(task, "simulator_backend", None))
    print("simulator_device:", getattr(task, "simulator_device", None))
    print("simulator_receiver_chunk:", getattr(task, "simulator_receiver_chunk", None))
    print("simulator_batch_chunk:", getattr(task, "simulator_batch_chunk", None))
    print("simulator_max_delays:", getattr(task, "simulator_max_delays", None))
    try:
        import torch

        print("torch.cuda.is_available:", bool(torch.cuda.is_available()))
        print("torch.backends.mps.is_available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
    except Exception:
        pass
    print("theta shape:", theta.shape, "dtype:", theta.dtype)
    print("x shape:", x.shape, "dtype:", x.dtype)
    print("x min/max:", float(np.min(x)), float(np.max(x)))
    print("obs shape:", x_obs.shape, "obs min/max:", float(np.min(x_obs)), float(np.max(x_obs)))
    print("theta seconds:", theta_s)
    print("sim seconds:", sim_s)
    print("total seconds:", total_s)
    print("sim throughput (samples/sec):", n / max(sim_s, 1e-12))

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = project_root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "benchmark_simulator_vs_obs.png"

        k = np.arange(ns)
        sims = x[:, :, 0]
        n_plot = int(min(int(args.plot_n), sims.shape[0]))

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i in range(n_plot):
            ax.plot(k, sims[i], color="#8ecae6", alpha=0.25, linewidth=1.0)
        ax.plot(k, x_obs[:ns], color="red", linewidth=3.0, label="Observed (turin_obs.npy)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Power (dB)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
