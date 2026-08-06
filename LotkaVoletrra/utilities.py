import torch
import numpy as np
import time
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# =============================================================================
# Shared Helpers
# =============================================================================

def fit_kde_and_evaluate(samples, evaluate_points):
    """
    Fit a KDE to samples and evaluate log density at evaluate_points.
    """
    # Use cross-validation or heuristic for bandwidth?
    # Simple heuristic: Scott's Rule
    n, d = samples.shape
    bandwidth = n ** (-1.0 / (d + 4))
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(samples)
    log_density = kde.score_samples(evaluate_points)
    return log_density


class BoundaryCorrectedKDE:
    """
    KDE on a bounded box via logit-transform to an unconstrained space.
    """

    def __init__(self, lower, upper, bandwidth=None, eps=1e-6):
        self.lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        self.bandwidth = bandwidth
        self.eps = float(eps)
        self.kde = None

    def _validate_input(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.lower.shape[0]:
            raise ValueError(
                f"Expected input with shape (n, {self.lower.shape[0]}), got {x.shape}"
            )
        return x

    def _transform(self, x):
        span = np.maximum(self.upper - self.lower, 1e-12)
        u = (x - self.lower) / span
        u = np.clip(u, self.eps, 1.0 - self.eps)
        z = np.log(u) - np.log1p(-u)
        log_abs_det = np.sum(-np.log(span) - np.log(u) - np.log1p(-u), axis=1)
        return z, log_abs_det

    def fit(self, samples):
        samples = self._validate_input(samples)
        z_samples, _ = self._transform(samples)
        if self.bandwidth is None:
            n, d = z_samples.shape
            bandwidth = float(n ** (-1.0 / (d + 4)))
        else:
            bandwidth = float(self.bandwidth)
        self.kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        self.kde.fit(z_samples)
        self.bandwidth_ = bandwidth
        return self

    def score_samples(self, evaluate_points):
        if self.kde is None:
            raise RuntimeError("BoundaryCorrectedKDE must be fit before calling score_samples")
        x = self._validate_input(evaluate_points)
        in_bounds = np.all((x >= self.lower) & (x <= self.upper), axis=1)
        log_density = np.full(x.shape[0], -np.inf, dtype=np.float64)
        if np.any(in_bounds):
            z_eval, log_abs_det = self._transform(x[in_bounds])
            log_density[in_bounds] = self.kde.score_samples(z_eval) + log_abs_det
        return log_density


def reflect_into_bounds(x, lower, upper):
    """
    Reflect proposals into a box [lower, upper] without hard clipping.
    """
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    span = np.maximum(upper - lower, 1e-12)
    doubled_span = 2.0 * span
    shifted = np.mod(x - lower, doubled_span)
    reflected = np.where(shifted <= span, lower + shifted, upper - (shifted - span))
    return reflected


def prepare_theta_for_log_space(theta, lower, upper, context, atol=1e-10):
    """
    Validate theta-space samples before taking logs.
    Reflection should keep proposals inside bounds; this only tolerates
    tiny floating-point drift at the boundary and raises on real violations.
    """
    theta = np.asarray(theta, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    below = theta < (lower - atol)
    above = theta > (upper + atol)
    if np.any(below | above):
        theta_min = np.min(theta, axis=0)
        theta_max = np.max(theta, axis=0)
        raise ValueError(
            f"{context}: theta out of bounds for theta-space MCMC. "
            f"min={theta_min}, max={theta_max}, lower={lower}, upper={upper}"
        )

    slightly_out = (theta < lower) | (theta > upper)
    if np.any(slightly_out):
        warnings.warn(
            f"{context}: theta had tiny boundary drift; snapping to bounds within atol={atol}.",
            RuntimeWarning,
            stacklevel=2,
        )
        theta = np.where(theta < lower, lower, theta)
        theta = np.where(theta > upper, upper, theta)

    return theta

def run_mcmc_refinement(current_theta, x_obs_stats, task,
                       log_prior_fn, log_likelihood_fn,
                       n_chains, n_samples, burn_in, thin,
                       proposal_std=0.5, lower=None, upper=None):
    """
    Generic MCMC refinement loop.
    """
    # Clip to bounds initially
    if lower is None:
        lower = task.lower
    if upper is None:
        upper = task.upper
    current_theta = reflect_into_bounds(current_theta, lower, upper)
    
    # Dimensional-wise Proposal Step Size
    std_per_dim = np.std(current_theta, axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    proposal_scale = std_per_dim * proposal_std
    print(f"Proposal Scale (per dim): {proposal_scale}")
    
    current_log_prior = log_prior_fn(current_theta)
    current_log_prob = np.full(n_chains, -np.inf, dtype=np.float64)
    valid_mask_curr = np.isfinite(current_log_prior)
    if np.any(valid_mask_curr):
        theta_valid_curr = current_theta[valid_mask_curr]
        log_lik_valid_curr = np.asarray(log_likelihood_fn(theta_valid_curr, x_obs_stats), dtype=np.float64)
        current_log_prob[valid_mask_curr] = current_log_prior[valid_mask_curr] + log_lik_valid_curr
    
    samples = []
    total_accepted_all = 0
    total_accepted_post_burn = 0
    total_reflected_all = 0
    start_time = time.time()
    
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps
    
    print(f"Total MCMC steps: {total_steps} (Burn-in: {burn_in}, Sampling: {total_sampling_steps})")
    
    for step in range(1, total_steps + 1):
        # Propose
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale
        proposed_theta_raw = current_theta + proposal_noise
        proposed_theta = reflect_into_bounds(proposed_theta_raw, lower, upper)
        reflected_mask = np.any(np.abs(proposed_theta - proposed_theta_raw) > 1e-12, axis=1)
        
        # Proposed Log Prob
        proposed_log_prior = log_prior_fn(proposed_theta)
        proposed_log_prob = np.full(n_chains, -np.inf, dtype=np.float64)
        valid_mask = np.isfinite(proposed_log_prior)
        if np.any(valid_mask):
            theta_valid = proposed_theta[valid_mask]
            log_lik_valid = np.asarray(log_likelihood_fn(theta_valid, x_obs_stats), dtype=np.float64)
            proposed_log_prob[valid_mask] = proposed_log_prior[valid_mask] + log_lik_valid
        
        log_ratio = proposed_log_prob - current_log_prob
        log_ratio = np.where(np.isfinite(log_ratio), log_ratio, -np.inf)
        log_u = np.log(np.random.rand(n_chains))
        accept_mask = log_u < log_ratio
        total_accepted_all += np.sum(accept_mask)
        total_reflected_all += np.sum(reflected_mask)
        
        # Update
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_log_prob[accept_mask] = proposed_log_prob[accept_mask]
        
        if step > burn_in:
            total_accepted_post_burn += np.sum(accept_mask)
            if (step - burn_in) % thin == 0:
                samples.append(current_theta.copy())
                
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(
                f"Step {step}/{total_steps} | Accepted: {np.mean(accept_mask):.2%} "
                f"| Reflected: {np.mean(reflected_mask):.2%} | Time: {elapsed:.2f}s"
            )
            
    if not samples:
        samples.append(current_theta.copy())
        
    posterior_samples = np.vstack(samples)
    acceptance_rate_all = total_accepted_all / (n_chains * total_steps) if total_steps > 0 else 0
    acceptance_rate_post_burn = (
        total_accepted_post_burn / (n_chains * total_sampling_steps)
        if total_sampling_steps > 0 else 0
    )
    reflection_rate_all = total_reflected_all / (n_chains * total_steps) if total_steps > 0 else 0
    print(
        "Refinement Complete. "
        f"Acceptance Rate (all steps): {acceptance_rate_all:.2%} | "
        f"Acceptance Rate (post-burn): {acceptance_rate_post_burn:.2%} | "
        f"Reflection Rate (all steps): {reflection_rate_all:.2%}"
    )
    
    return posterior_samples

def save_summary_distance_distribution(distances, epsilon, quantile_level, output_path):
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if distances.size == 0 or output_path is None:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distances, bins=min(60, max(20, distances.size // 20)), color="#4C72B0", alpha=0.8, density=True)
    ax.axvline(float(epsilon), color="#C44E52", linestyle="--", linewidth=2.5, label=f"epsilon @ q={quantile_level:.3f}")
    ax.set_title("Summary Distance Distribution")
    ax.set_xlabel("||S(x) - S(x*)||")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_bandwidth_core(
    theta0_np,
    x_obs_tensor,
    task,
    stats_fn,
    n_samples,
    quantile_level,
    device,
    preprocess_obs_fn=None,
    plot_path=None,
):
    """
    Core bandwidth computation logic.
    """
    # Simulate Data from Theta0
    # Note: task.simulator in LV returns (batch, n_obs, 2)
    theta0_np = np.clip(theta0_np, task.lower, task.upper)
    xn_0 = task.simulator(theta0_np, n_samples=task.n_obs)
    if preprocess_obs_fn is not None:
        xn_0 = preprocess_obs_fn(xn_0)
    xn_0_tensor = torch.from_numpy(xn_0).float().to(device)
    
    with torch.no_grad():
        TT = stats_fn(xn_0_tensor).detach().cpu().to(torch.float64)
        T_target = stats_fn(x_obs_tensor).detach().cpu().to(torch.float64)
        
        diff = T_target - TT
        dist_sq = torch.sum(diff**2, dim=1)
        dist = torch.sqrt(dist_sq)
        
    dist_np = dist.cpu().numpy()
    quan1 = float(max(np.quantile(dist_np, quantile_level), 1e-12))
    print(f"Computed bandwidth (epsilon): {quan1}")
    if plot_path is not None:
        save_summary_distance_distribution(dist_np, quan1, quantile_level, plot_path)
        print(f"Saved summary distance distribution plot to: {plot_path}")
    return quan1

def approximate_likelihood_core(
    theta,
    x_obs_stats,
    task,
    stats_fn,
    epsilon,
    device,
    preprocess_obs_fn=None,
    n_likelihood_replicates=5,
):
    """
    Core approximate log-likelihood computation.
    Returns log(1/K * sum_i kernel(||S(x_i)-S(x*)|| / epsilon)) in float64.
    """
    theta = np.asarray(theta)
    if theta.ndim != 2:
        raise ValueError(f"Expected theta with shape (batch, d), got {theta.shape}")

    n_reps = max(int(n_likelihood_replicates), 1)

    # 1. Simulate (Numpy) - forced by task.simulator
    # theta: (batch, d)
    theta = np.clip(theta, task.lower, task.upper)
    theta_repeated = np.repeat(theta, n_reps, axis=0)
    sim_data = task.simulator(theta_repeated, n_samples=task.n_obs)
    if preprocess_obs_fn is not None:
        sim_data = preprocess_obs_fn(sim_data)
    
    # 2. To Tensor
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    
    with torch.no_grad():
        # 3. Stats (Tensor)
        sim_stats = stats_fn(sim_data_tensor).detach().cpu().to(torch.float64)
        
        # 4. Distance (Tensor)
        # x_obs_stats should be (1, dim) or (dim,)
        # sim_stats is (batch, dim)
        diff = sim_stats - x_obs_stats.detach().cpu().to(torch.float64)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        # 5. Kernel (Gaussian), but stay in log-space for numerical stability.
        log_kernel = -dist_sq / (2.0 * float(epsilon) ** 2)
        log_kernel = log_kernel.reshape(theta.shape[0], n_reps)
        log_likelihood = torch.logsumexp(log_kernel, dim=1) - np.log(n_reps)
    
    return log_likelihood.cpu().numpy()

# =============================================================================
# SMMD/MMD Implementation
# =============================================================================

def compute_bandwidth_smmd(
    model,
    x_obs,
    task,
    n_samples=5000,
    quantile_level=0.01,
    device=None,
    preprocess_obs_fn=None,
    plot_path=None,
):
    if device is None:
        # Try to infer device from model parameters
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device("cpu")
        
    print(f"Computing bandwidth for SMMD refinement (Device: {device})...")
    
    # Ensure x_obs is float32 and on correct device
    # Handle x_obs if it's already a tensor or numpy
    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)
        
    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)
        
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1)
    
    with torch.no_grad():
        stats = model.T(x_obs_batch)
        Z = torch.randn(n_samples, 1, task.d, device=device)
        Theta0 = model.unscale_theta(model.G(Z, stats)).squeeze(1)
    
    Theta0_np = Theta0.cpu().numpy()
    
    return compute_bandwidth_core(
        Theta0_np,
        x_obs_tensor,
        task,
        model.T,
        n_samples,
        quantile_level,
        device,
        preprocess_obs_fn=preprocess_obs_fn,
        plot_path=plot_path,
    )

def refine_posterior_smmd(model, x_obs, task, 
                         n_chains=1000, n_samples=1, burn_in=99, 
                         thin=1, epsilon=None, proposal_std=0.5, device=None, theta_init=None,
                         bandwidth_n_samples=5000, bandwidth_quantile_level=0.01,
                         preprocess_obs_fn=None, n_likelihood_replicates=5, bandwidth_plot_path=None):
    if device is None:
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device("cpu")
            
    print(f"Starting SMMD MCMC Refinement (Device: {device})...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_smmd(
            model,
            x_obs,
            task,
            n_samples=bandwidth_n_samples,
            quantile_level=bandwidth_quantile_level,
            device=device,
            preprocess_obs_fn=preprocess_obs_fn,
            plot_path=bandwidth_plot_path,
        )
        
    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)
        
    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)
    
    # 1. Stats and Initial Samples
    theta_lower = np.exp(np.asarray(task.lower, dtype=np.float64))
    theta_upper = np.exp(np.asarray(task.upper, dtype=np.float64))

    with torch.no_grad():
        x_obs_stats = model.T(x_obs_tensor)
        # Initial samples
        if theta_init is not None:
            print(f"Using provided initial points for MCMC (shape: {theta_init.shape})")
            current_theta = np.asarray(theta_init, dtype=np.float64)
            # Ensure shape matches n_chains if possible, or just use what is given
            if current_theta.shape[0] != n_chains:
                print(f"Warning: theta_init shape {current_theta.shape} != n_chains {n_chains}. Adjusting n_chains.")
                n_chains = current_theta.shape[0]
        else:
            Z = torch.randn(n_chains, 1, task.d, device=device)
            current_theta_log = model.unscale_theta(
                model.G(Z, x_obs_stats.expand(n_chains, -1))
            ).squeeze(1).cpu().numpy()
            current_theta = np.exp(np.clip(current_theta_log, -50.0, 50.0))
        
    # 2. Define Likelihood Function
    def log_likelihood_fn(theta, target_stats):
        theta_checked = prepare_theta_for_log_space(
            theta, theta_lower, theta_upper, context="SMMD log_likelihood_fn"
        )
        theta_log = np.log(theta_checked)
        return approximate_likelihood_core(
            theta_log,
            target_stats,
            task,
            model.T,
            epsilon,
            device,
            preprocess_obs_fn=preprocess_obs_fn,
            n_likelihood_replicates=n_likelihood_replicates,
        )

    def log_prior_fn(theta):
        theta_checked = prepare_theta_for_log_space(
            theta, theta_lower, theta_upper, context="SMMD log_prior_fn"
        )
        theta_log = np.log(theta_checked)
        log_prior_log_space = np.asarray(task.log_prior(theta_log), dtype=np.float64)
        # MCMC now runs in theta-space, so the target density needs the
        # change-of-variables term from log-theta to theta.
        log_abs_det_dlogtheta_dtheta = -np.sum(np.log(theta_checked), axis=1)
        return log_prior_log_space + log_abs_det_dlogtheta_dtheta
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              log_prior_fn, log_likelihood_fn,
                              n_chains, n_samples, burn_in, thin, proposal_std,
                              lower=theta_lower, upper=theta_upper)

# =============================================================================
# BayesFlow Implementation
# =============================================================================

def refine_posterior_bayesflow(model, x_obs, task, 
                             n_chains=1000, n_samples=1, burn_in=99, 
                             thin=1, epsilon=None, proposal_std=0.5, device=None, theta_init=None,
                             bandwidth_n_samples=5000, bandwidth_quantile_level=0.01,
                             preprocess_obs_fn=None, n_likelihood_replicates=5, bandwidth_plot_path=None):
    if device is None:
        # Infer device from model (Keras model usually manages its own device, but we need it for tensors)
        device = "cpu"
        print("Warning: Device not provided for BayesFlow refinement. Defaulting to CPU.")

    print("Starting BayesFlow MCMC Refinement...")
    theta_lower = np.exp(np.asarray(task.lower, dtype=np.float64))
    theta_upper = np.exp(np.asarray(task.upper, dtype=np.float64))
    
    # 1. Bandwidth
    if epsilon is None:
        epsilon = compute_bandwidth_bayesflow(
            model,
            x_obs,
            task,
            n_samples=bandwidth_n_samples,
            quantile_level=bandwidth_quantile_level,
            device=device,
            preprocess_obs_fn=preprocess_obs_fn,
            plot_path=bandwidth_plot_path,
        )
        
    # 2. Initial Points
    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]

    if theta_init is not None:
        print(f"Using provided initial points for MCMC (shape: {theta_init.shape})")
        current_theta = np.asarray(theta_init, dtype=np.float64)
        if current_theta.shape[0] != n_chains:
             print(f"Warning: theta_init shape {current_theta.shape} != n_chains {n_chains}. Adjusting n_chains.")
             n_chains = current_theta.shape[0]
    else:
        x_obs_rep = np.tile(x_obs_cpu, (n_chains, 1, 1))
        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if isinstance(post, torch.Tensor):
            post = post.detach().cpu().numpy()
        if hasattr(post, "numpy"):
            post = post.numpy()
        current_theta_log = np.asarray(post).reshape(n_chains, -1)
        current_theta = np.exp(np.clip(current_theta_log, -50.0, 50.0))
        
    # 3. MCMC
    x_obs_tensor = torch.from_numpy(x_obs_cpu).float().to(device)
    # BayesFlow Summary Network
    # We need to wrap it to accept torch tensors and return torch tensors
    # Keras models usually expect numpy or TF tensors.
    # But approximate_likelihood_core expects a callable that takes torch tensors.
    
    def stats_fn_torch(x_tensor):
        # x_tensor: (batch, n_obs, d_x)
        x_np = x_tensor.detach().cpu().numpy().astype(np.float32)
        stats = model.summary_network(x_np)
        if isinstance(stats, torch.Tensor):
            stats_np = stats.detach().cpu().numpy()
        elif hasattr(stats, "numpy"):
            stats_np = stats.numpy()
        else:
            stats_np = np.asarray(stats)
        return torch.from_numpy(np.asarray(stats_np, dtype=np.float32)).float().to(device)

    # Pre-compute x_obs_stats
    # model.summary_network might return a tensor or numpy
    # Ensure float32 for model input
    x_obs_cpu = x_obs_cpu.astype(np.float32)
    x_obs_stats_tf = model.summary_network(x_obs_cpu)
    if isinstance(x_obs_stats_tf, torch.Tensor):
        x_obs_stats_np = x_obs_stats_tf.detach().cpu().numpy()
    elif hasattr(x_obs_stats_tf, "numpy"):
        x_obs_stats_np = x_obs_stats_tf.numpy()
    else:
        x_obs_stats_np = np.asarray(x_obs_stats_tf)
    x_obs_stats = torch.from_numpy(np.asarray(x_obs_stats_np, dtype=np.float32)).float().to(device)
    
    def log_likelihood_fn(theta, target_stats):
        theta_checked = prepare_theta_for_log_space(
            theta, theta_lower, theta_upper, context="BayesFlow log_likelihood_fn"
        )
        theta_log = np.log(theta_checked)
        return approximate_likelihood_core(
            theta_log,
            target_stats,
            task,
            stats_fn_torch,
            epsilon,
            device,
            preprocess_obs_fn=preprocess_obs_fn,
            n_likelihood_replicates=n_likelihood_replicates,
        )

    def log_prior_fn(theta):
        theta_checked = prepare_theta_for_log_space(
            theta, theta_lower, theta_upper, context="BayesFlow log_prior_fn"
        )
        theta_log = np.log(theta_checked)
        log_prior_log_space = np.asarray(task.log_prior(theta_log), dtype=np.float64)
        log_abs_det_dlogtheta_dtheta = -np.sum(np.log(theta_checked), axis=1)
        return log_prior_log_space + log_abs_det_dlogtheta_dtheta
        
    return run_mcmc_refinement(current_theta, x_obs_stats, task,
                              log_prior_fn, log_likelihood_fn,
                              n_chains, n_samples, burn_in, thin, proposal_std,
                              lower=theta_lower, upper=theta_upper)

def compute_bandwidth_bayesflow(
    model,
    x_obs,
    task,
    n_samples=5000,
    quantile_level=0.01,
    device="cpu",
    preprocess_obs_fn=None,
    plot_path=None,
):
    print(f"Computing bandwidth for BayesFlow refinement...")

    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
    x_obs_cpu = x_obs_cpu.astype(np.float32)

    x_obs_tensor = torch.from_numpy(x_obs_cpu).float().to(device)

    def stats_fn_torch(x_tensor):
        x_np = x_tensor.detach().cpu().numpy().astype(np.float32)
        stats = model.summary_network(x_np)
        if isinstance(stats, torch.Tensor):
            stats_np = stats.detach().cpu().numpy()
        elif hasattr(stats, "numpy"):
            stats_np = stats.numpy()
        else:
            stats_np = np.asarray(stats)
        return torch.from_numpy(np.asarray(stats_np, dtype=np.float32)).float().to(device)

    theta0_np = None
    try:
        x_obs_rep = np.tile(x_obs_cpu, (n_samples, 1, 1))
        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if hasattr(post, "numpy"):
            post = post.numpy()
        theta0_np = np.asarray(post).reshape(n_samples, -1)
    except Exception as e:
        print(f"Warning: Failed to sample BayesFlow posterior for bandwidth: {e}")

    if theta0_np is None:
        theta0_np = task.sample_prior(n_samples, "vague")

    return compute_bandwidth_core(
        theta0_np,
        x_obs_tensor,
        task,
        stats_fn_torch,
        n_samples,
        quantile_level,
        device,
        preprocess_obs_fn=preprocess_obs_fn,
        plot_path=plot_path,
    )

# Backwards compatibility / Dispatcher
def refine_posterior(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return refine_posterior_bayesflow(model, *args, **kwargs)
    else:
        return refine_posterior_smmd(model, *args, **kwargs)

def compute_bandwidth_torch(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return compute_bandwidth_bayesflow(model, *args, **kwargs)
    else:
        return compute_bandwidth_smmd(model, *args, **kwargs)

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(samples, theta_true, sample_space="log"):
    """
    Compute Bias, 95% HDI, and Coverage.
    
    Args:
        samples: (n_samples, d) approximate posterior samples
        theta_true: (d,) ground truth parameters
    
    Returns:
        dict with keys:
            'bias': (d,) L2 distance between mean and true
            'hdi_lower': (d,) 2.5% quantile
            'hdi_upper': (d,) 97.5% quantile
            'hdi_length': (d,) upper - lower
            'coverage': (d,) 1 if true in HDI, 0 otherwise (as float)
    """
    # 1. Bias (Mean - True)
    # L2 distance between mean and true parameters.
    
    samples = np.asarray(samples, dtype=np.float64)
    theta_true = np.asarray(theta_true, dtype=np.float64)

    if sample_space == "log":
        samples_exp = np.exp(np.clip(samples, -50.0, 50.0))
    elif sample_space == "theta":
        samples_exp = np.clip(samples, 1e-12, None)
    else:
        raise ValueError(f"Unknown sample_space: {sample_space}")
    theta_true_exp = np.exp(theta_true)

    mean_est = np.mean(samples_exp, axis=0)
    bias_vec = mean_est - theta_true_exp
    bias_l2 = np.linalg.norm(bias_vec) # Scalar L2 distance
    
    # 2. 95% HDI (Credible Interval)
    # Using quantiles [0.025, 0.975]
    lower = np.quantile(samples_exp, 0.025, axis=0)
    upper = np.quantile(samples_exp, 0.975, axis=0)
    length = upper - lower
    
    # 3. Coverage
    # Check if theta_true is within [lower, upper]
    covered = (theta_true_exp >= lower) & (theta_true_exp <= upper)
    
    return {
        "bias_l2": bias_l2,
        "bias_vec": bias_vec,
        "hdi_lower": lower,
        "hdi_upper": upper,
        "hdi_length": length,
        "coverage": covered.astype(float)
    }


def calibration_error_from_rounds(
    posterior_samples_by_round,
    theta_true_by_round,
    n_steps=100,
    posterior_sample_space="log",
):
    posterior_samples_by_round = np.asarray(posterior_samples_by_round, dtype=np.float64)
    theta_true_by_round = np.asarray(theta_true_by_round, dtype=np.float64)

    if posterior_samples_by_round.ndim != 3:
        raise ValueError(
            f"Expected posterior_samples_by_round with shape (n_rounds, n_samples, d), "
            f"got {posterior_samples_by_round.shape}"
        )
    if theta_true_by_round.ndim != 2:
        raise ValueError(
            f"Expected theta_true_by_round with shape (n_rounds, d), got {theta_true_by_round.shape}"
        )

    n_rounds, _, d_param = posterior_samples_by_round.shape
    if theta_true_by_round.shape[0] != n_rounds or theta_true_by_round.shape[1] != d_param:
        raise ValueError(
            f"Shape mismatch: posterior_samples_by_round is {posterior_samples_by_round.shape}, "
            f"theta_true_by_round is {theta_true_by_round.shape}"
        )

    if posterior_sample_space == "log":
        posterior_samples_by_round = np.exp(np.clip(posterior_samples_by_round, -50.0, 50.0))
    elif posterior_sample_space == "theta":
        posterior_samples_by_round = np.clip(posterior_samples_by_round, 1e-12, None)
    else:
        raise ValueError(f"Unknown posterior_sample_space: {posterior_sample_space}")
    theta_true_by_round = np.exp(theta_true_by_round)

    confidences = np.linspace(0.0, 1.0, n_steps + 1, endpoint=False, dtype=np.float32)[1:]
    q_values = []
    for conf in confidences:
        q_low = 0.5 * (1.0 - float(conf))
        q_high = 0.5 * (1.0 + float(conf))
        q_values.extend([q_low, q_high])

    quantiles = np.quantile(posterior_samples_by_round, q_values, axis=1)
    inlier_fractions = np.zeros((n_steps, d_param), dtype=np.float32)
    median_interval_lengths = np.zeros((n_steps, d_param), dtype=np.float32)

    for i in range(n_steps):
        low = quantiles[2 * i]
        high = quantiles[2 * i + 1]
        inliers = (theta_true_by_round > low) & (theta_true_by_round < high)
        inlier_fractions[i] = np.mean(inliers.astype(np.float32), axis=0)
        median_interval_lengths[i] = np.median((high - low).astype(np.float32), axis=0)

    calib_err = inlier_fractions - confidences[:, None]
    median_abs_per_dim = np.median(np.abs(calib_err), axis=0)
    median_abs_mean = float(np.mean(median_abs_per_dim))

    return {
        "confidences": confidences,
        "inlier_fractions": inlier_fractions,
        "median_interval_lengths": median_interval_lengths,
        "calibration_error": calib_err,
        "median_abs_calibration_error_per_dim": median_abs_per_dim.astype(np.float32),
        "median_abs_calibration_error_mean": median_abs_mean,
        "n_rounds": int(n_rounds),
        "n_steps": int(n_steps),
    }
