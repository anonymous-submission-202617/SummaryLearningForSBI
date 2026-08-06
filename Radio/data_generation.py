import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _simulate_turin_single_sample(theta: np.ndarray, Ns: int, N_rx: int, B: float, tau0: float) -> np.ndarray:
    G0, T, lambda_0, sigma2_N = theta.astype(np.float64)
    delta_f = B / (Ns - 1)
    t_max = 1.0 / delta_f
    k = np.arange(Ns, dtype=np.float64)
    k_phase = (-1j * 2.0 * np.pi * delta_f) * k[:, None]
    Y_all = np.zeros((N_rx, Ns), dtype=np.complex128)
    mu_poisson = max(lambda_0, 0.0) * t_max
    if mu_poisson <= 0:
        mu_poisson = 1e-9
    for j in range(N_rx):
        n_points = np.random.poisson(mu_poisson)
        if n_points > 0:
            delays = np.sort(np.random.uniform(0.0, t_max, size=(n_points,)))
            active = delays >= float(tau0)
            sigma2 = (G0 * np.exp(-delays / T) / lambda_0) * B
            std = np.sqrt(np.maximum(sigma2, 0.0) / 2.0)
            std = np.maximum(std, 1e-7)
            std = np.where(active, std, 0.0)
            alpha = np.random.normal(0.0, std) + 1j * np.random.normal(0.0, std)
            exp_mat = np.exp(k_phase @ delays[None, :])
            H = (exp_mat @ alpha).astype(np.complex128)
        else:
            H = np.zeros((Ns,), dtype=np.complex128)
        noise_std = np.sqrt(max(sigma2_N, 0.0) / 2.0)
        noise_std = max(noise_std, 1e-7)
        W = np.random.normal(0.0, noise_std, size=(Ns,)) + 1j * np.random.normal(0.0, noise_std, size=(Ns,))
        Y_all[j, :] = H + W
    y_time = np.fft.ifft(Y_all, axis=1)
    power = np.abs(y_time) ** 2
    p_db = 10.0 * np.log10(power)
    return p_db.astype(np.float32)


def _simulate_turin_batch_torch(
    theta: np.ndarray,
    Ns: int,
    N_rx: int,
    B: float,
    *,
    tau0: float,
) -> np.ndarray:
    import torch

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=dev)
    theta_t = theta_t if theta_t.ndim == 2 else theta_t.unsqueeze(0)

    bsz = theta_t.shape[0]
    delta_f = float(B) / float(Ns - 1)
    t_max = float(Ns - 1) / float(B)

    k = torch.arange(Ns, device=dev, dtype=torch.float32)
    omega = (2.0 * np.pi * float(delta_f)) * k

    out = torch.empty((bsz, int(N_rx), Ns), device=dev, dtype=torch.float32)

    batch_chunk = int(os.environ.get("RADIO_SIM_BATCH_CHUNK", "32"))
    receiver_chunk = int(os.environ.get("RADIO_SIM_RECEIVER_CHUNK", "8"))
    delay_chunk = int(os.environ.get("RADIO_SIM_DELAY_CHUNK", "64"))
    max_n_paths = int(os.environ.get("RADIO_SIM_MAX_PATHS", "8192"))

    for b0 in range(0, bsz, int(batch_chunk)):
        b1 = min(bsz, b0 + int(batch_chunk))
        th = theta_t[b0:b1]
        g0 = th[:, 0].view(-1, 1, 1)
        t0 = torch.clamp(th[:, 1].view(-1, 1, 1), min=1e-12)
        lam0 = torch.clamp(th[:, 2].view(-1, 1, 1), min=1e-12)
        sig2n = th[:, 3].view(-1, 1, 1)

        mu_poisson = torch.clamp(lam0.view(-1), min=0.0) * float(t_max)
        mu_poisson = torch.where(torch.isfinite(mu_poisson), mu_poisson, torch.zeros_like(mu_poisson))
        mu_poisson = torch.clamp(mu_poisson, min=1e-9, max=float(max_n_paths))

        out_chunk = torch.empty((b1 - b0, int(N_rx), Ns), device=dev, dtype=torch.float32)

        for r0 in range(0, int(N_rx), int(receiver_chunk)):
            r1 = min(int(N_rx), r0 + int(receiver_chunk))
            rchunk = r1 - r0

            mu_expanded = mu_poisson[:, None].expand(-1, rchunk)
            if dev.type == "mps":
                n_points = torch.poisson(mu_expanded.detach().cpu()).to(torch.int64).to(dev)
            else:
                n_points = torch.poisson(mu_expanded).to(torch.int64)
            n_points = torch.clamp(n_points, min=0, max=int(max_n_paths))
            nmax = int(n_points.max().item()) if ((b1 - b0) > 0 and rchunk > 0) else 0

            m = (b1 - b0) * rchunk
            if nmax == 0:
                h = torch.zeros((m, Ns), dtype=torch.cfloat, device=dev)
            else:
                h = torch.zeros((m, Ns), dtype=torch.cfloat, device=dev)
                for d0 in range(0, nmax, int(delay_chunk)):
                    d1 = min(nmax, d0 + int(delay_chunk))
                    k_blk = d1 - d0
                    delays_blk = torch.rand((b1 - b0, rchunk, k_blk), device=dev, dtype=torch.float32) * float(t_max)
                    idx_blk = torch.arange(d0, d1, device=dev, dtype=torch.int64).view(1, 1, k_blk)
                    valid_blk = idx_blk < n_points[:, :, None]
                    active_blk = valid_blk & (delays_blk >= float(tau0))

                    sigma2_blk = (g0 * torch.exp(-delays_blk / t0) / lam0) * float(B)
                    std_blk = torch.sqrt(torch.clamp(sigma2_blk / 2.0, min=0.0))
                    std_blk = torch.clamp(std_blk, min=1e-7)
                    std_blk = torch.where(active_blk, std_blk, torch.zeros_like(std_blk))
                    alpha_blk = torch.complex(torch.randn_like(std_blk), torch.randn_like(std_blk)) * std_blk

                    delays_flat = delays_blk.reshape(m, k_blk)
                    alpha_flat = alpha_blk.reshape(m, k_blk)
                    phase = omega.view(1, Ns, 1) * delays_flat.view(m, 1, k_blk)
                    exp_blk = torch.exp(torch.complex(torch.zeros_like(phase), -phase))
                    h = h + torch.bmm(exp_blk, alpha_flat.unsqueeze(-1)).squeeze(-1)

            stdN = torch.sqrt(torch.clamp(sig2n / 2.0, min=0.0))
            stdN = torch.where(stdN > 0.0, stdN, torch.full_like(stdN, 1e-7))
            noise_std = stdN.view(-1, 1).expand(-1, rchunk).reshape(m, 1)
            w = torch.complex(torch.randn((m, Ns), device=dev), torch.randn((m, Ns), device=dev)) * noise_std

            y = torch.fft.ifft(h + w, dim=1)
            power = torch.abs(y) ** 2
            p_db = 10.0 * torch.log10(power)

            out_chunk[:, r0:r1, :] = p_db.view(b1 - b0, rchunk, Ns)

        out[b0:b1] = out_chunk

    return out.detach().cpu().numpy().astype(np.float32)


@dataclass(frozen=True)
class RadioTask:
    Ns: int = 801
    N_rx: int = 100
    B: float = 4e9
    tau0: float = 6e-9
    config_path: Optional[str] = None
    prior_lower: Optional[Sequence[float]] = None
    prior_upper: Optional[Sequence[float]] = None

    def __post_init__(self):
        object.__setattr__(self, "d", 4)
        object.__setattr__(self, "d_x", 1)
        cfg = {}
        cfg_path = self.config_path or os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(cfg_path):
            cfg = load_config(cfg_path)

        lower_raw = self.prior_lower if self.prior_lower is not None else cfg.get("prior_lower")
        upper_raw = self.prior_upper if self.prior_upper is not None else cfg.get("prior_upper")

        if lower_raw is None:
            lower_raw = [1e-10, 1e-10, 1e7, 1e-11]
        if upper_raw is None:
            upper_raw = [1e-7, 1e-7, 5e9, 1e-8]

        lower = np.asarray(lower_raw, dtype=np.float32).reshape(4)
        upper = np.asarray(upper_raw, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(upper > lower):
            raise ValueError(f"Invalid prior bounds. lower={lower}, upper={upper}")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def sample_prior(self, batch_size: int) -> np.ndarray:
        bsz = int(batch_size)
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        if np.any(lower <= 0.0) or np.any(upper <= 0.0):
            raise ValueError(f"Log-uniform prior requires positive bounds. lower={lower}, upper={upper}")
        log_lower = np.log(lower, dtype=np.float32)
        log_upper = np.log(upper, dtype=np.float32)
        u = np.random.rand(bsz, 4).astype(np.float32)
        log_theta = log_lower + u * (log_upper - log_lower)
        theta = np.exp(log_theta, dtype=np.float32)
        return theta.astype(np.float32)

    def theta_mean(self) -> np.ndarray:
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        log_lower = np.log(lower, dtype=np.float32)
        log_upper = np.log(upper, dtype=np.float32)
        denom = np.maximum(log_upper - log_lower, 1e-30).astype(np.float32)
        return ((upper - lower) / denom).astype(np.float32)

    def theta_std(self) -> np.ndarray:
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        log_lower = np.log(lower, dtype=np.float32)
        log_upper = np.log(upper, dtype=np.float32)
        denom = np.maximum(log_upper - log_lower, 1e-30).astype(np.float32)
        e1 = ((upper - lower) / denom).astype(np.float32)
        e2 = ((upper * upper - lower * lower) / (2.0 * denom)).astype(np.float32)
        var = np.maximum(e2 - e1 * e1, 0.0).astype(np.float32)
        return np.sqrt(var, dtype=np.float32)

    def zscore_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        m = self.theta_mean()
        s = self.theta_std()
        return ((theta - m) / s).astype(np.float32)

    def inv_zscore_theta(self, theta_z: np.ndarray) -> np.ndarray:
        theta_z = np.asarray(theta_z, dtype=np.float32)
        m = self.theta_mean()
        s = self.theta_std()
        return (theta_z * s + m).astype(np.float32)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        in_bounds = np.all((theta >= lower) & (theta <= upper), axis=1) & np.all(theta > 0.0, axis=1)
        logp = np.full(theta.shape[0], -np.inf, dtype=np.float32)
        if np.any(in_bounds):
            log_lower = np.log(lower, dtype=np.float32)
            log_upper = np.log(upper, dtype=np.float32)
            widths = np.maximum(log_upper - log_lower, 1e-30).astype(np.float32)
            const = -np.sum(np.log(widths, dtype=np.float32)).astype(np.float32)
            logp[in_bounds] = const - np.sum(np.log(theta[in_bounds], dtype=np.float32), axis=1).astype(np.float32)
        if logp.shape[0] == 1:
            return np.float32(logp[0])
        return logp

    def simulator(self, theta: np.ndarray, n_rx: Optional[int] = None) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        theta = np.atleast_2d(theta)
        n_rx_eff = int(n_rx) if n_rx is not None else int(self.N_rx)
        return _simulate_turin_batch_torch(theta, self.Ns, n_rx_eff, self.B, tau0=float(self.tau0))


def load_task(project_root: str, config_path: Optional[str] = None) -> Tuple[RadioTask, np.ndarray]:
    cfg_path = config_path or os.path.join(project_root, "config.json")
    cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
    ns = int(cfg.get("Ns", 801))
    n_rx = int(cfg.get("N_rx", 100))
    b = float(cfg.get("B", 4e9))
    tau0 = float(cfg.get("tau0", 6e-9))
    prior_lower = cfg.get("prior_lower")
    prior_upper = cfg.get("prior_upper")
    task = RadioTask(
        Ns=ns,
        N_rx=n_rx,
        B=b,
        tau0=tau0,
        config_path=cfg_path,
        prior_lower=prior_lower,
        prior_upper=prior_upper,
    )
    obs_path = os.path.join(project_root, "data", "turin_obs.npy")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Missing observed data at {obs_path}")
    obs = np.load(obs_path).astype(np.float32)
    if obs.ndim == 3:
        obs = obs.reshape(obs.shape[-2], obs.shape[-1])
    if obs.ndim == 1 and obs.size % task.Ns == 0:
        obs = obs.reshape(-1, task.Ns)
    if obs.ndim != 2:
        obs = obs.reshape(-1)
        obs = obs[: int(task.N_rx) * int(task.Ns)].reshape(int(task.N_rx), int(task.Ns))
    obs = obs[: int(task.N_rx), : int(task.Ns)]
    return task, obs.astype(np.float32)
