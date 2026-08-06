import time
from typing import Callable

import numpy as np
import torch


def run_mcmc_refinement(
    current_theta: np.ndarray,
    x_obs_stats: torch.Tensor,
    task,
    log_prior_fn: Callable[[np.ndarray], np.ndarray],
    likelihood_fn: Callable[[np.ndarray, torch.Tensor], np.ndarray],
    n_chains: int,
    n_samples: int,
    burn_in: int,
    thin: int,
    proposal_std: float = 0.5,
) -> np.ndarray:
    # --- Convert to φ = log θ space ---
    # Prior is log-uniform → φ = log θ is bounded uniform.
    # MCMC runs in φ-space with symmetric Gaussian proposal + reflection at bounds,
    # then converts to θ = exp(φ) for simulator / prior / likelihood evaluation.
    current_theta = np.asarray(current_theta, dtype=np.float32)
    current_theta = np.clip(current_theta, task.lower, task.upper)
    phi_current = np.log(current_theta).astype(np.float32)

    log_lower = np.log(task.lower).astype(np.float32)
    log_upper = np.log(task.upper).astype(np.float32)
    phi_range = log_upper - log_lower

    # --- Compute proposal scale in φ-space ---
    std_per_dim = np.std(phi_current, axis=0).astype(np.float32)
    min_scale = 0.05 * phi_range
    std_eff = np.maximum(std_per_dim, min_scale)
    proposal_scale = std_eff * float(proposal_std)
    proposal_scale = np.minimum(proposal_scale, 0.5 * phi_range)
    print(f"[Refine] Proposal scale per dim (log-space): {proposal_scale}")
    print(f"[Refine] Starting ABC-MCMC (log-space) with n_chains={n_chains}, burn_in={burn_in}, n_samples={n_samples}, thin={thin}")

    # --- Compute initial log-probability in φ-space ---
    # Reparameterisation: φ = log θ,  Jacobian |det dθ/dφ| = Π exp(φᵢ) = exp(Σ φᵢ)
    # log π_φ(φ) = log π_θ(exp(φ)) + Σ φᵢ
    log_jac_current = np.sum(phi_current, axis=1).astype(np.float32)
    current_log_prior = log_prior_fn(current_theta)
    current_log_prob = np.full(n_chains, -np.inf, dtype=np.float32)
    valid_mask_curr = np.isfinite(current_log_prior)
    if np.any(valid_mask_curr):
        theta_valid_curr = current_theta[valid_mask_curr]
        lik_valid_curr = likelihood_fn(theta_valid_curr, x_obs_stats)
        log_lik_valid_curr = np.log(lik_valid_curr + 1e-300)
        current_log_prob[valid_mask_curr] = (
            current_log_prior[valid_mask_curr]
            + log_lik_valid_curr.astype(np.float32)
            + log_jac_current[valid_mask_curr]
        )
    else:
        print("[Refine] WARNING: All initial theta points have zero prior probability!")

    samples = []
    total_accepted = 0
    total_accepted_all = 0
    total_sampling_steps = int(n_samples) * int(thin)
    total_steps = int(burn_in) + total_sampling_steps

    t_start = time.time()
    for step in range(1, total_steps + 1):
        # --- Propose in φ-space (symmetric Gaussian, reflected at bounds) ---
        proposal_noise = np.random.randn(n_chains, task.d).astype(np.float32) * proposal_scale
        phi_proposed = phi_current + proposal_noise

        # Reflect at log-space bounds
        for d in range(task.d):
            below = phi_proposed[:, d] < log_lower[d]
            if np.any(below):
                phi_proposed[below, d] = 2.0 * log_lower[d] - phi_proposed[below, d]
            above = phi_proposed[:, d] > log_upper[d]
            if np.any(above):
                phi_proposed[above, d] = 2.0 * log_upper[d] - phi_proposed[above, d]

        # --- Convert to θ-space for simulator / prior / likelihood ---
        proposed_theta = np.exp(phi_proposed).astype(np.float32)

        # --- Compute proposed log-probability in φ-space ---
        log_jac_proposed = np.sum(phi_proposed, axis=1).astype(np.float32)
        proposed_log_prior = log_prior_fn(proposed_theta)
        proposed_log_prob = np.full(n_chains, -np.inf, dtype=np.float32)
        valid_mask = np.isfinite(proposed_log_prior)
        if np.any(valid_mask):
            theta_valid = proposed_theta[valid_mask]
            lik_valid = likelihood_fn(theta_valid, x_obs_stats)
            log_lik_valid = np.log(lik_valid + 1e-300)
            proposed_log_prob[valid_mask] = (
                proposed_log_prior[valid_mask]
                + log_lik_valid.astype(np.float32)
                + log_jac_proposed[valid_mask]
            )

        # --- MH acceptance ---
        log_ratio = proposed_log_prob - current_log_prob
        log_ratio = np.where(np.isfinite(log_ratio), log_ratio, -np.inf).astype(np.float32)
        log_u = np.log(np.random.rand(n_chains).astype(np.float32) + 1e-12)
        accept_mask = log_u < log_ratio
        total_accepted_all += int(np.sum(accept_mask))

        phi_current[accept_mask] = phi_proposed[accept_mask]
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_log_prob[accept_mask] = proposed_log_prob[accept_mask]

        if step > burn_in:
            total_accepted += int(np.sum(accept_mask))
            if (step - burn_in) % thin == 0:
                samples.append(current_theta.copy())

        if step == burn_in:
            burn_in_acc_rate = total_accepted_all / float(n_chains * burn_in) if burn_in > 0 else 0.0
            print(
                f"[Refine] Finished burn-in at step={step}, "
                f"burn-in acceptance rate={burn_in_acc_rate:.3f}"
            )
        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            elapsed = time.time() - t_start
            if step > burn_in:
                eff_steps = step - burn_in
                sampling_acc_rate = total_accepted / float(n_chains * eff_steps) if eff_steps > 0 else 0.0
                phase_label = f"sampling rate={sampling_acc_rate:.3f}"
            else:
                burn_in_rate_now = total_accepted_all / float(n_chains * step) if step > 0 else 0.0
                phase_label = f"burn-in rate={burn_in_rate_now:.3f}"
            print(
                f"[Refine] Step {step}/{total_steps}, accepted this step={float(np.mean(accept_mask)):.3f}, "
                f"valid_prop={float(np.mean(valid_mask)):.3f}, {phase_label}, elapsed={elapsed:.1f}s"
            )

    if not samples:
        samples.append(current_theta.copy())
    total_sampling_steps_effective = int(n_samples) * int(thin)
    acceptance_rate = total_accepted / float(n_chains * total_sampling_steps_effective) if total_sampling_steps_effective > 0 else 0.0
    acceptance_rate_all = total_accepted_all / float(n_chains * total_steps) if total_steps > 0 else 0.0
    print(
        f"[Refine] Finished ABC-MCMC (log-space): total_steps={total_steps}, "
        f"sampling_acceptance_rate={acceptance_rate:.3f}, "
        f"overall_acceptance_rate={acceptance_rate_all:.3f}, "
        f"num_samples={np.vstack(samples).shape[0]}"
    )
    return np.vstack(samples)


def compute_bandwidth_core(theta0_np, x_obs_tensor, task, stats_fn, n_samples, quantile_level, device):
    xn_0 = task.simulator(theta0_np)
    xn_0_tensor = torch.from_numpy(xn_0).float().to(device)
    with torch.no_grad():
        TT = stats_fn(xn_0_tensor)
        T_target = stats_fn(x_obs_tensor)
        diff = T_target - TT
        dist_sq = torch.sum(diff * diff, dim=1)
    dist_sq_np = dist_sq.detach().cpu().numpy()
    return float(np.quantile(dist_sq_np, quantile_level))


def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, epsilon, device):
    theta = np.asarray(theta, dtype=np.float32)
    sim_data = task.simulator(theta)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    with torch.no_grad():
        sim_stats = stats_fn(sim_data_tensor)
        diff = sim_stats - x_obs_stats
        dist_sq = torch.sum(diff * diff, dim=-1)
        likelihood = torch.exp(-dist_sq / (2.0 * float(epsilon)))
    return likelihood.detach().cpu().numpy()

def _destandardize_theta_if_needed(model, theta: np.ndarray) -> np.ndarray:
    if hasattr(model, "theta_mean_buf") and hasattr(model, "theta_std_buf"):
        try:
            if int(model.theta_mean_buf.numel()) > 0 and int(model.theta_std_buf.numel()) > 0:
                tm = model.theta_mean_buf.detach().cpu().numpy().astype(np.float32)
                ts = model.theta_std_buf.detach().cpu().numpy().astype(np.float32)
                return theta.astype(np.float32) * ts + tm
        except Exception:
            pass
    return theta


def _bf_to_numpy(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    if hasattr(v, "cpu") and hasattr(v, "numpy"):
        try:
            return v.cpu().numpy()
        except Exception:
            pass
    if hasattr(v, "numpy"):
        return v.numpy()
    return np.asarray(v)


def _ensure_valid_theta(theta: np.ndarray, task, max_rounds: int = 20) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim == 1:
        theta = theta[None, :]
    for _ in range(int(max_rounds)):
        valid = np.isfinite(task.log_prior(theta))
        if np.all(valid):
            return theta
        n_bad = int(np.sum(~valid))
        theta[~valid] = task.sample_prior(n_bad).astype(np.float32)
    valid = np.isfinite(task.log_prior(theta))
    return theta[valid]


def compute_bandwidth_bayesflow(
    model,
    x_obs: np.ndarray,
    task,
    *,
    n_samples: int = 2000,
    quantile_level: float = 0.02,
) -> float:
    theta_prior = task.sample_prior(int(n_samples)).astype(np.float32)
    x_sim = task.simulator(theta_prior).astype(np.float32)

    x_obs_np = np.asarray(x_obs, dtype=np.float32)
    if x_obs_np.ndim == 1:
        x_obs_np = x_obs_np[None, None, :]
    if x_obs_np.ndim == 2:
        x_obs_np = x_obs_np[None, :, :]

    stats_sim = _bf_to_numpy(model.summary_network(np.asarray(x_sim, dtype=np.float32))).astype(np.float32)
    stats_obs = _bf_to_numpy(model.summary_network(np.asarray(x_obs_np, dtype=np.float32))).astype(np.float32).reshape(-1)
    diff = stats_sim - stats_obs[None, :]
    dist_sq = np.sum(diff * diff, axis=1)
    return float(np.quantile(dist_sq, float(quantile_level)))


def refine_posterior_bayesflow(
    model,
    x_obs,
    task,
    *,
    n_chains: int = 1000,
    n_samples: int = 1,
    burn_in: int = 49,
    thin: int = 1,
    epsilon: float | None = None,
    proposal_std: float = 0.2,
    epsilon_quantile: float = 0.02,
    bandwidth_n_samples: int = 2000,
    device=None,
) -> np.ndarray:
    print("[Refine] Using BayesFlow refine_posterior_bayesflow")
    if device is None:
        device = torch.device("cpu")

    x_obs_np = np.asarray(x_obs, dtype=np.float32)
    if x_obs_np.ndim == 1:
        x_obs_np = x_obs_np[None, None, :]
    if x_obs_np.ndim == 2:
        x_obs_np = x_obs_np[None, :, :]

    stats_obs_np = _bf_to_numpy(model.summary_network(np.asarray(x_obs_np, dtype=np.float32))).astype(np.float32)
    x_obs_stats = torch.from_numpy(stats_obs_np).float().to(device)

    if epsilon is None:
        epsilon = compute_bandwidth_bayesflow(
            model,
            x_obs_np,
            task,
            n_samples=int(bandwidth_n_samples),
            quantile_level=float(epsilon_quantile),
        )
        print(f"[Refine] Estimated epsilon={epsilon:.6f} (quantile={epsilon_quantile}, n={bandwidth_n_samples})")

    x_obs_rep = np.repeat(np.asarray(x_obs_np, dtype=np.float32), repeats=int(n_chains), axis=0)
    post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
    if isinstance(post, dict):
        post = post.get("inference_variables", post)
    current_theta = _bf_to_numpy(post).astype(np.float32).reshape(int(n_chains), -1)
    current_theta = _ensure_valid_theta(current_theta, task)
    print(f"[Refine] Initial theta std per dim: {np.std(current_theta, axis=0)}")

    target_stats_np = x_obs_stats.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def likelihood_fn(theta_np: np.ndarray, _target_stats: torch.Tensor) -> np.ndarray:
        theta_np = np.asarray(theta_np, dtype=np.float32)
        x_sim = task.simulator(theta_np).astype(np.float32)
        stats_sim = _bf_to_numpy(model.summary_network(np.asarray(x_sim, dtype=np.float32))).astype(np.float32)
        diff = stats_sim - target_stats_np[None, :]
        dist_sq = np.sum(diff * diff, axis=1)
        lik = np.exp(-dist_sq / (2.0 * float(epsilon))).astype(np.float32)
        return lik

    return run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        likelihood_fn,
        int(n_chains),
        int(n_samples),
        int(burn_in),
        int(thin),
        float(proposal_std),
    )


def refine_posterior_smmd(
    model,
    x_obs,
    task,
    n_chains=1000,
    n_samples=1,
    burn_in=49,
    thin=1,
    epsilon=None,
    proposal_std=0.2,
    device=None,
    epsilon_quantile=0.02,
    bandwidth_n_samples=2000,
):
    print("[Refine] Using SMMD/MMD refine_posterior_smmd")
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    x_obs_np = np.asarray(x_obs, dtype=np.float32)
    if x_obs_np.ndim == 1:
        x_obs_tensor = torch.from_numpy(x_obs_np[None, :, None]).float().to(device)
    elif x_obs_np.ndim == 2:
        x_obs_tensor = torch.from_numpy(x_obs_np[None, :, :]).float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs_np).float().to(device)

    stats_fn = model.compute_stats if hasattr(model, "compute_stats") else model.T

    if epsilon is None:
        theta0_np = task.sample_prior(int(bandwidth_n_samples)).astype(np.float32)
        epsilon = compute_bandwidth_core(
            theta0_np,
            x_obs_tensor,
            task,
            stats_fn,
            int(bandwidth_n_samples),
            float(epsilon_quantile),
            device,
        )
        print(f"[Refine] Estimated epsilon={epsilon:.6f} (quantile={epsilon_quantile}, n={bandwidth_n_samples})")

    with torch.no_grad():
        x_obs_stats = stats_fn(x_obs_tensor)
        z = torch.randn(int(n_chains), 1, task.d, device=device)
        theta0 = model.G(z, x_obs_stats.expand(int(n_chains), -1)).squeeze(1).detach().cpu().numpy().astype(np.float32)
        current_theta = _destandardize_theta_if_needed(model, theta0)
        current_theta = _ensure_valid_theta(current_theta, task)
        print(f"[Refine] Initial theta std per dim: {np.std(current_theta, axis=0)}")
    def likelihood_fn(theta_np, target_stats):
        return approximate_likelihood_core(theta_np, target_stats, task, stats_fn, epsilon, device)
    return run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        likelihood_fn,
        int(n_chains),
        int(n_samples),
        int(burn_in),
        int(thin),
        float(proposal_std),
    )


def turin_temporal_moments_from_pdb(y_freq_like: np.ndarray, *, K: int = 4, B: float = 4e9) -> np.ndarray:
    """
    Reference-style Turin summary statistic used by the original MMD evaluation.

    This intentionally mirrors `reference/mmd_metric.py`:
    1. apply `np.fft.ifft` to each realization;
    2. compute temporal moments of `|ifft(y)|^2`;
    3. return the log-moments.

    Notes:
    - The current Radio pipeline stores observations/simulations as real arrays, but
      for consistency with the reference metric we still apply `ifft` directly here.
    - Input shape is `(N, Ns)` or `(Ns,)`.
    """
    y_freq_like = np.asarray(y_freq_like)
    if y_freq_like.ndim == 1:
        y_freq_like = y_freq_like[None, :]
    if y_freq_like.ndim != 2:
        raise ValueError(f"Expected input with shape (N, Ns) or (Ns,), got {y_freq_like.shape}")

    n, ns = y_freq_like.shape
    delta_f = float(B) / float(ns - 1)
    t_max = 1.0 / delta_f
    tau = np.linspace(0.0, t_max, ns, dtype=np.float64)

    y_time = np.fft.ifft(y_freq_like, axis=1)
    power = np.abs(y_time) ** 2
    tau_pow = np.stack([tau ** int(k) for k in range(int(K))], axis=0)
    integrand = power[None, :, :] * tau_pow[:, None, :]
    moments = np.trapezoid(integrand, x=tau, axis=-1)
    moments = np.maximum(moments, 1e-300)
    return np.log(moments).T.astype(np.float32, copy=False)


def mmd_unweighted_rbf(x: np.ndarray, y: np.ndarray, *, lengthscale: float = 1.0, device=None) -> float:
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if x_t.ndim != 2 or y_t.ndim != 2:
        raise ValueError(f"Expected x,y with shape (N,D). Got x={tuple(x_t.shape)}, y={tuple(y_t.shape)}")
    if x_t.shape[1] != y_t.shape[1]:
        raise ValueError(f"Feature dim mismatch: x has D={x_t.shape[1]}, y has D={y_t.shape[1]}")

    if device is None:
        device = x_t.device
    x_t = x_t.to(device)
    y_t = y_t.to(device)

    m = int(x_t.shape[0])
    n = int(y_t.shape[0])
    z = torch.cat((x_t, y_t), dim=0)
    d2 = torch.cdist(z, z) ** 2
    l = float(lengthscale)
    K = torch.exp(-(1.0 / (2.0 * l * l)) * d2)

    kxx = K[:m, :m]
    kyy = K[m : m + n, m : m + n]
    kxy = K[:m, m : m + n]

    out = (1.0 / (m * m)) * torch.sum(kxx) - (2.0 / (m * n)) * torch.sum(kxy) + (1.0 / (n * n)) * torch.sum(kyy)
    return float(out.detach().cpu().item())
