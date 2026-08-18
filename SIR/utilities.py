import torch
import numpy as np
import time
from sklearn.neighbors import KernelDensity


def _rejection_sample_theta(theta_array, n_needed, rng, label="theta"):
    """
    Filter theta to finite, positive, non-extreme draws; bootstrap if short.

    This protects downstream simulator / MCMC calls from NaN, θ ≤ 0, or
    extreme values that SMMD/MMD/BayesFlow generators can occasionally produce
    (their output layers have no positivity constraint).
    """
    valid = np.all(np.isfinite(theta_array) & (theta_array > 0) & (theta_array < 100.0), axis=1)
    n_valid = np.sum(valid)
    if n_valid == 0:
        raise RuntimeError(f"[Refine] All {label} samples are invalid — model may be degenerate.")
    if n_valid < n_needed:
        if n_valid < n_needed * 0.1:
            print(
                f"[Refine] WARNING: only {n_valid}/{len(theta_array)} {label} "
                f"samples valid (< 10%), model output may need attention"
            )
        idx = rng.choice(n_valid, size=n_needed, replace=True)
    else:
        idx = rng.choice(n_valid, size=n_needed, replace=False)
    return theta_array[valid][idx]


def _draw_smmd_samples(model, x_obs_stats, n_raw, device, d):
    """Draw *n_raw* samples from an SMMD/MMD generator."""
    with torch.no_grad():
        z = torch.randn(n_raw, 1, d, device=device)
        return model.G(z, x_obs_stats.expand(n_raw, -1)).squeeze(1).cpu().numpy()


def _draw_bayesflow_samples(model, x_obs_cpu, n_raw):
    """Draw *n_raw* samples from a BayesFlow amortised posterior."""
    x_rep = np.tile(x_obs_cpu.astype(np.float32), (n_raw, 1, 1))
    post = model.sample(conditions={"summary_variables": x_rep}, num_samples=1)
    if isinstance(post, dict):
        post = post["inference_variables"]
    if hasattr(post, "numpy"):
        post = post.numpy()
    return np.asarray(post).reshape(n_raw, -1)


def sample_posterior_valid(model, x_context, n_needed, device, d,
                           is_bayesflow=False, max_attempts=3, label="posterior"):
    """
    Draw *n_needed* valid posterior samples via rejection + iterative resampling.

    SMMD/MMD/BayesFlow inference networks may output θ ≤ 0, NaN, or extreme
    values.  This routine draws extra samples each round, keeps only valid ones,
    and retries until enough are collected (or bootstraps from the valid pool on
    the last attempt).
    """
    # Seed from the global numpy stream so that rejection sampling is
    # reproducible whenever the caller seeds ``np.random``.
    rng = np.random.RandomState(np.random.randint(0, 2**31 - 1))
    pool = None
    for attempt in range(1, max_attempts + 1):
        # Oversample: draw 2× the remaining deficit (at least n_needed)
        remaining = n_needed - (len(pool) if pool is not None else 0)
        n_draw = max(n_needed, remaining * 2)
        if is_bayesflow:
            theta = _draw_bayesflow_samples(model, x_context, n_draw)
        else:
            theta = _draw_smmd_samples(model, x_context, n_draw, device, d)

        valid = np.all(np.isfinite(theta) & (theta > 0) & (theta < 100.0), axis=1)
        new = theta[valid]
        pool = np.concatenate([pool, new], axis=0) if pool is not None else new
        if len(pool) >= n_needed:
            break
    else:
        print(
            f"[Refine] WARNING: only {len(pool)}/{n_needed} valid {label} "
            f"samples after {max_attempts} attempts, bootstrapping"
        )

    return _rejection_sample_theta(pool, n_needed, rng, label)


def fit_kde_and_evaluate(samples, evaluate_points):
    n, d = samples.shape
    bandwidth = n ** (-1.0 / (d + 4))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(samples)
    log_density = kde.score_samples(evaluate_points)
    return log_density


def run_mcmc_refinement(
    current_theta,
    x_obs_stats,
    task,
    log_prior_fn,
    log_likelihood_fn,
    n_chains,
    n_samples,
    burn_in,
    thin,
    proposal_std=0.5,
):
    """Run ABC-MCMC refinement in log-theta space (phi = log theta).

    Args:
        current_theta: initial theta for each chain (n_chains, d).
        x_obs_stats: summary statistics of the observed data.
        task: SIR task providing ``log_prior`` and simulator access.
        log_prior_fn: function mapping theta -> log prior density.
        log_likelihood_fn: function mapping (theta, x_obs_stats) -> log ABC
            kernel value (already in log space; no exp/log round-trip).
        n_chains: number of parallel chains.
        n_samples: number of posterior samples kept per chain.
        burn_in: number of burn-in steps.
        thin: thinning interval during the sampling phase.
        proposal_std: proposal scale multiplier in phi-space.

    Returns:
        Posterior samples with shape (n_chains * n_samples, d).
    """
    current_theta = np.maximum(current_theta, task.lower)  # safety floor for log(θ) > -∞

    # Replace NaN / inf chains with median of valid chains so they don't
    # permanently stall (NaN never accepts in MH and wastes a chain).
    nan_mask = ~np.all(np.isfinite(current_theta), axis=1)
    if np.any(nan_mask):
        valid = current_theta[~nan_mask]
        fill_val = np.median(valid, axis=0) if len(valid) > 0 else task.lower * 10.0
        current_theta[nan_mask] = fill_val
        print(
            f"[Refine] WARNING: Replaced {np.sum(nan_mask)} / {n_chains} "
            f"NaN chains with median of valid chains"
        )

    # --- Reparameterization: work in log-space (φ = log θ) ---
    # Because SIR parameters (beta, gamma) are LogNormal-distributed and
    # strictly positive, the posterior is typically more Gaussian in log-space.
    # Proposing in φ-space with a Jacobian correction avoids wasted proposals
    # that cross the θ ≥ 0 boundary and improves chain mixing.
    phi_current = np.log(current_theta)
    log_jac_current = np.sum(phi_current, axis=1)  # log |det dθ/dφ| = Σ φ

    # Proposal step-size calibrated in φ-space (based on amortised posterior spread)
    std_per_dim_phi = np.std(phi_current, axis=0)
    std_per_dim_phi = np.maximum(std_per_dim_phi, 1e-6)
    proposal_scale_phi = std_per_dim_phi * proposal_std
    # ------------------------------------------------------------------

    current_log_prior = log_prior_fn(current_theta)

    # Initialize current log_prob with -inf
    current_log_prob = np.full(n_chains, -np.inf)

    # Check for valid priors (finite log prior + reasonable range)
    valid_mask_curr = np.isfinite(current_log_prior) & np.all(current_theta < 100.0, axis=1)

    if np.any(valid_mask_curr):
        theta_valid_curr = current_theta[valid_mask_curr]
        log_lik_valid_curr = log_likelihood_fn(theta_valid_curr, x_obs_stats)
        current_log_prob[valid_mask_curr] = current_log_prior[valid_mask_curr] + log_lik_valid_curr
    else:
        # If all initial points are invalid, warn but don't crash
        print("[Refine] WARNING: All initial theta points have zero prior probability!")

    print(
        f"[Refine] Starting ABC-MCMC (log-space reparameterisation) with n_chains={n_chains}, "
        f"burn_in={burn_in}, n_samples={n_samples}, thin={thin}"
    )
    print(f"[Refine] Proposal std per dim (log-space): {proposal_scale_phi}")
    samples = []
    total_accepted_all = 0
    total_accepted_post_burn = 0
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps

    t_start = time.time()
    for step in range(1, total_steps + 1):
        # 1. Propose in φ-space (unbounded Gaussian random walk)
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale_phi
        phi_proposed = phi_current + proposal_noise

        # 2. Map back to θ-space for log-prior / likelihood evaluation
        proposed_theta = np.exp(phi_proposed)

        proposed_log_prior = log_prior_fn(proposed_theta)

        # Initialize log_prob with -inf
        proposed_log_prob = np.full(n_chains, -np.inf)

        # Check for valid priors (finite log prior) and reasonable range
        # (extreme values like theta > 100 can cause the SIR ODE to diverge)
        valid_mask = np.isfinite(proposed_log_prior) & np.all(proposed_theta < 100.0, axis=1)

        # Compute likelihood only for valid proposals to avoid simulator crashes
        if np.any(valid_mask):
            # Extract valid parameters
            theta_valid = proposed_theta[valid_mask]

            # Compute log-likelihood (already in log space)
            log_lik_valid = log_likelihood_fn(theta_valid, x_obs_stats)

            # Update log_prob for valid entries
            proposed_log_prob[valid_mask] = proposed_log_prior[valid_mask] + log_lik_valid

        # 3. Jacobian correction: log |det dθ/dφ| = Σ φ
        log_jac_proposed = np.sum(phi_proposed, axis=1)

        # 4. MH acceptance in φ-space target:
        #    log π_φ(φ) = log π_θ(exp(φ)) + Σ φ
        #    Suppress invalid-value warning from -inf - (-inf) on dead chains.
        with np.errstate(invalid="ignore"):
            log_ratio = (proposed_log_prob + log_jac_proposed) - (current_log_prob + log_jac_current)
        log_u = np.log(np.random.rand(n_chains))
        accept_mask = log_u < log_ratio
        step_accepted = np.sum(accept_mask)
        total_accepted_all += step_accepted
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_log_prob[accept_mask] = proposed_log_prob[accept_mask]
        phi_current[accept_mask] = phi_proposed[accept_mask]
        log_jac_current[accept_mask] = log_jac_proposed[accept_mask]
        if step > burn_in:
            total_accepted_post_burn += step_accepted
            if (step - burn_in) % thin == 0:
                samples.append(current_theta.copy())
        if step == burn_in:
            print(f"[Refine] Finished burn-in at step={step}")
        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            elapsed = time.time() - t_start
            acc_this_step = np.mean(accept_mask)
            acc_rate_all = total_accepted_all / (n_chains * step) if step > 0 else 0.0
            print(
                f"[Refine] Step {step}/{total_steps}, "
                f"accepted this step={acc_this_step:.2%}, "
                f"cumulative={acc_rate_all:.3f}, "
                f"elapsed={elapsed:.1f}s"
            )

    if not samples:
        samples.append(current_theta.copy())

    posterior_samples = np.vstack(samples)
    acc_rate_all = (
        total_accepted_all / (n_chains * total_steps)
        if total_steps > 0
        else 0.0
    )
    acc_rate_post_burn = (
        total_accepted_post_burn / (n_chains * total_sampling_steps)
        if total_sampling_steps > 0
        else 0.0
    )
    print(
        f"[Refine] Finished ABC-MCMC: total_steps={total_steps}, "
        f"acceptance (all steps)={acc_rate_all:.3f}, "
        f"acceptance (post-burn)={acc_rate_post_burn:.3f}, "
        f"num_samples={posterior_samples.shape[0]}"
    )
    return posterior_samples


def compute_bandwidth_core(
    theta0_np, x_obs_tensor, task, stats_fn, quantile_level, device
):
    xn_0 = task.simulator(theta0_np, n_samples=task.n_obs)
    xn_0_tensor = torch.from_numpy(xn_0).float().to(device)

    with torch.no_grad():
        TT = stats_fn(xn_0_tensor)
        T_target = stats_fn(x_obs_tensor)
        diff = T_target - TT
        dist_sq = torch.sum(diff**2, dim=1)
        dist = torch.sqrt(dist_sq)

    dist_np = dist.cpu().numpy()
    quan1 = np.quantile(dist_np, quantile_level)
    return quan1


def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, epsilon, device):
    """ABC kernel log-likelihood: ``-||T(x) - T(x_obs)||^2 / (2 eps^2)``.

    This is the single-dataset version of the approximate likelihood of
    Algorithm 1 (n_sims = 1 for the SIR task): one dataset X ~ P_theta^(n) is
    simulated per likelihood evaluation. Returned directly in log space
    (computed in float64) so that distant proposals cannot underflow to zero;
    the MCMC refinement consumes these log values directly.
    """
    sim_data = task.simulator(theta, n_samples=task.n_obs)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)

    with torch.no_grad():
        sim_stats = stats_fn(sim_data_tensor)
        diff = sim_stats.double() - x_obs_stats.double()
        dist_sq = torch.sum(diff**2, dim=-1)
        log_likelihood = -dist_sq / (2 * epsilon**2)

    return log_likelihood.cpu().numpy()


def compute_metrics(theta_samples, theta_true, ci=0.95):
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.detach().cpu().numpy()
    theta_true = np.asarray(theta_true, dtype=np.float32)
    if theta_true.ndim == 1:
        theta_true = theta_true[np.newaxis, :]
    theta_true = theta_true[0]

    mean_est = np.mean(theta_samples, axis=0)
    bias_l2 = np.linalg.norm(mean_est - theta_true)
    rmse = float(np.sqrt(np.mean((theta_samples - theta_true) ** 2)))

    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    hdi_lower = np.quantile(theta_samples, lower_q, axis=0)
    hdi_upper = np.quantile(theta_samples, upper_q, axis=0)
    hdi_length = hdi_upper - hdi_lower

    coverage = (theta_true >= hdi_lower) & (theta_true <= hdi_upper)
    coverage = coverage.astype(np.float32)

    return {
        "bias_l2": bias_l2,
        "rmse": rmse,
        "hdi_length": hdi_length,
        "coverage": coverage,
    }


def refine_posterior_smmd(
    model,
    x_obs,
    task,
    n_chains=1000,
    n_samples=1,
    burn_in=99,
    thin=1,
    epsilon=None,
    proposal_std=0.5,
    device=None,
    stats_fn=None,
    bandwidth_n_samples=5000,
):
    print("[Refine] Using SMMD/MMD refine_posterior_smmd")
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)

    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)

    if stats_fn is None:
        stats_fn = model.T

    # Compute observed summary statistics once (reused below)
    with torch.no_grad():
        x_obs_stats = stats_fn(x_obs_tensor)

    if epsilon is None:
        print(
            f"[Refine] Estimating epsilon with {bandwidth_n_samples} "
            f"approximate posterior samples for d={task.d}"
        )
        theta_post_for_eps = sample_posterior_valid(
            model, x_obs_stats, bandwidth_n_samples, device, task.d,
            is_bayesflow=False, label="epsilon"
        )

        epsilon = compute_bandwidth_core(
            theta_post_for_eps,
            x_obs_tensor,
            task,
            stats_fn,
            0.01,
            device,
        )
        print(f"[Refine] Estimated epsilon={epsilon:.4f}")

    # Initialise MCMC chains from the amortised posterior
    current_theta = sample_posterior_valid(
        model, x_obs_stats, n_chains, device, task.d,
        is_bayesflow=False, label="MCMC init"
    )
    print(
        f"[Refine] Initial theta for MCMC has shape={current_theta.shape}, "
        f"std_per_dim={np.std(current_theta, axis=0)}"
    )

    def log_likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, stats_fn, epsilon, device)

    samples = run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        log_likelihood_fn,
        n_chains,
        n_samples,
        burn_in,
        thin,
        proposal_std,
    )
    print(f"[Refine] Returned {samples.shape[0]} refined samples from SMMD/MMD")
    return samples


def refine_posterior_bayesflow(
    model,
    x_obs,
    task,
    n_chains=1000,
    n_samples=1,
    burn_in=99,
    thin=1,
    epsilon=None,
    proposal_std=0.5,
    device=None,
    theta_init=None,
    bandwidth_n_samples=5000,
):
    print("[Refine] Using BayesFlow refine_posterior_bayesflow")
    if device is None:
        device = "cpu"

    if epsilon is None:
        print(
            f"[Refine] Estimating epsilon for BayesFlow with "
            f"{bandwidth_n_samples} approximate posterior samples"
        )
        epsilon = compute_bandwidth_bayesflow(
            model, x_obs, task, n_samples=bandwidth_n_samples, device=device
        )
        print(f"[Refine] Estimated epsilon={epsilon:.4f}")

    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[np.newaxis, ...]

    if theta_init is not None:
        current_theta = theta_init
        if current_theta.shape[0] != n_chains:
            n_chains = current_theta.shape[0]
    else:
        current_theta = sample_posterior_valid(
            model, x_obs_cpu, n_chains, device=None, d=task.d,
            is_bayesflow=True, label="MCMC init"
        )
    print(
        f"[Refine] Initial theta for BayesFlow MCMC has shape={current_theta.shape}, "
        f"std_per_dim={np.std(current_theta, axis=0)}"
    )

    def stats_fn_torch(x_tensor):
        x_np = x_tensor.cpu().numpy().astype(np.float32)
        stats = model.summary_network(x_np)
        if hasattr(stats, "cpu"):
            stats = stats.cpu()
        if hasattr(stats, "numpy"):
            stats = stats.numpy()
        return torch.from_numpy(stats).float().to(device)

    x_obs_cpu = x_obs_cpu.astype(np.float32)
    x_obs_stats_tf = model.summary_network(x_obs_cpu)
    if hasattr(x_obs_stats_tf, "cpu"):
        x_obs_stats_tf = x_obs_stats_tf.cpu()
    if hasattr(x_obs_stats_tf, "numpy"):
        x_obs_stats_tf = x_obs_stats_tf.numpy()
    x_obs_stats = torch.from_numpy(x_obs_stats_tf).float().to(device)

    def log_likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(
            theta, target_stats, task, stats_fn_torch, epsilon, device
        )

    samples = run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        log_likelihood_fn,
        n_chains,
        n_samples,
        burn_in,
        thin,
        proposal_std,
    )
    print(f"[Refine] Returned {samples.shape[0]} refined samples from BayesFlow")
    return samples


def compute_bandwidth_bayesflow(
    model, x_obs, task, n_samples=5000, quantile_level=0.01, device="cpu"
):
    # Sample from approximate posterior (not prior) so epsilon is calibrated
    # to the region the MCMC actually explores
    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[np.newaxis, ...]
    theta_post = sample_posterior_valid(
        model, x_obs_cpu, n_samples, device=None, d=task.d,
        is_bayesflow=True, label="epsilon"
    )

    x_sim = task.simulator(theta_post)
    x_sim_float = x_sim.astype(np.float32)
    stats_sim = model.summary_network(x_sim_float)
    if hasattr(stats_sim, "cpu"):
        stats_sim = stats_sim.cpu()
    if hasattr(stats_sim, "numpy"):
        stats_sim = stats_sim.numpy()

    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[np.newaxis, ...]
    x_obs_cpu = x_obs_cpu.astype(np.float32)
    stats_obs = model.summary_network(x_obs_cpu)
    if hasattr(stats_obs, "cpu"):
        stats_obs = stats_obs.cpu()
    if hasattr(stats_obs, "numpy"):
        stats_obs = stats_obs.numpy()

    diff = stats_sim - stats_obs
    dist = np.linalg.norm(diff, axis=1)
    epsilon = np.quantile(dist, quantile_level)
    return epsilon


def refine_posterior(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return refine_posterior_bayesflow(model, *args, **kwargs)
    return refine_posterior_smmd(model, *args, **kwargs)


def compute_bandwidth_torch(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return compute_bandwidth_bayesflow(model, *args, **kwargs)
    return compute_bandwidth_core(model, *args, **kwargs)


def compute_calibration_from_rounds(theta_true_list, posterior_samples_list, n_steps=100):
    """
    Compute parameter-wise calibration error from per-round (theta_true, posterior) pairs.

    Each round provides one (theta_true, posterior_samples) tuple.  Across *R* rounds
    they are stacked into arrays of shape (R, samples_per_round, d) and (R, d), then
    processed identically to ``compute_expected_coverage``.

    Args:
        theta_true_list: list of (d,) arrays, one per round.
        posterior_samples_list: list of (n_samples, d) arrays, one per round.
        n_steps: number of equally-spaced credibility levels in (0, 1].

    Returns:
        dict with keys:
            confidences: (n_steps,) float32 array
            empirical_coverage: (n_steps, d) float32 array
            median_interval_lengths: (n_steps, d) float32 array
            calibration_error: (n_steps, d) float32 array  (= empirical_coverage - q)
            median_abs_calibration_error_per_dim: (d,) float32 array
            median_abs_calibration_error_mean: float
    """
    posterior_stack = np.stack(
        [np.asarray(s, dtype=np.float32) for s in posterior_samples_list], axis=0
    )  # (n_rounds, n_samples, d)
    theta_stack = np.stack(
        [np.asarray(t, dtype=np.float32) for t in theta_true_list], axis=0
    )  # (n_rounds, d)

    if posterior_stack.ndim != 3:
        raise ValueError(
            "posterior_samples_list must stack to (n_rounds, n_samples, d), "
            f"got {posterior_stack.shape}"
        )
    if theta_stack.ndim != 2:
        raise ValueError(
            "theta_true_list must stack to (n_rounds, d), "
            f"got {theta_stack.shape}"
        )

    confidences = np.linspace(0.0, 1.0, int(n_steps) + 1, endpoint=False, dtype=np.float32)[1:]

    # --- build quantile levels for all confidences at once ---
    q_values = []
    for conf in confidences:
        q_low = 0.5 * (1.0 - float(conf))
        q_high = 0.5 * (1.0 + float(conf))
        q_values.extend([q_low, q_high])

    quantiles = np.quantile(posterior_stack, q_values, axis=1)  # (2*n_steps, n_rounds, d)
    # reshape → (n_steps, 2, n_rounds, d)
    quantiles = quantiles.reshape(len(confidences), 2, posterior_stack.shape[0], posterior_stack.shape[2])

    empirical_coverage = np.zeros((len(confidences), theta_stack.shape[1]), dtype=np.float32)
    median_interval_lengths = np.zeros_like(empirical_coverage)

    for idx, _ in enumerate(confidences):
        low = quantiles[idx, 0]   # (n_rounds, d)
        high = quantiles[idx, 1]  # (n_rounds, d)
        inliers = (theta_stack > low) & (theta_stack < high)
        empirical_coverage[idx] = np.mean(inliers.astype(np.float32), axis=0)
        median_interval_lengths[idx] = np.median((high - low).astype(np.float32), axis=0)

    calibration_error = empirical_coverage - confidences[:, None]

    return {
        "confidences": confidences,
        "empirical_coverage": empirical_coverage,
        "median_interval_lengths": median_interval_lengths,
        "calibration_error": calibration_error,
        "median_abs_calibration_error_per_dim": np.median(np.abs(calibration_error), axis=0).astype(np.float32),
        "median_abs_calibration_error_mean": float(np.mean(np.median(np.abs(calibration_error), axis=0))),
    }
