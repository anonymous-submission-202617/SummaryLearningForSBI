import torch
import numpy as np
import time
import warnings

# =============================================================================
# Shared Helpers
# =============================================================================

def reflect_into_bounds(x, lower, upper):
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    span = np.maximum(upper - lower, 1e-12)
    doubled_span = 2.0 * span
    shifted = np.mod(x - lower, doubled_span)
    return np.where(shifted <= span, lower + shifted, upper - (shifted - span))

def reflect_into_bounds_masked(x, lower, upper, bounded_mask):
    x = np.asarray(x, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    bounded_mask = np.asarray(bounded_mask, dtype=bool).reshape(-1)
    if bounded_mask.size != lower.size:
        raise ValueError("bounded_mask must have the same length as lower/upper")
    if not np.any(bounded_mask):
        return x
    x_ref = x.copy()
    x_ref[:, bounded_mask] = reflect_into_bounds(
        x_ref[:, bounded_mask],
        lower[bounded_mask],
        upper[bounded_mask],
    )
    return x_ref


def prepare_theta_in_bounds(theta, lower, upper, context, atol=1e-10):
    theta = np.asarray(theta, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    below = theta < (lower - atol)
    above = theta > (upper + atol)
    if np.any(below | above):
        theta_min = np.min(theta, axis=0)
        theta_max = np.max(theta, axis=0)
        raise ValueError(
            f"{context}: theta out of bounds. min={theta_min}, max={theta_max}, lower={lower}, upper={upper}"
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

def propose_mixed_random_walk(current_theta, proposal_scale, lower, upper, bounded_mask):
    current_theta = np.asarray(current_theta, dtype=np.float64)
    proposal_scale = np.asarray(proposal_scale, dtype=np.float64).reshape(1, -1)
    noise = np.random.randn(current_theta.shape[0], current_theta.shape[1]) * proposal_scale
    proposed_raw = current_theta + noise
    proposed = reflect_into_bounds_masked(proposed_raw, lower, upper, bounded_mask)
    reflected_mask = np.zeros(current_theta.shape[0], dtype=bool)
    if np.any(bounded_mask):
        reflected_mask = np.any(
            np.abs(proposed[:, bounded_mask] - proposed_raw[:, bounded_mask]) > 1e-12,
            axis=1,
        )
    return proposed_raw, proposed, reflected_mask


def run_mcmc_refinement(current_theta, x_obs_stats, task, 
                       log_prior_fn, log_likelihood_fn, 
                       n_chains, n_samples, burn_in, thin, 
                       proposal_std=0.2):
    """
    Generic MCMC refinement loop (random-walk Metropolis--Hastings in theta space).

    Implements the Parallel MCMC step of Algorithm 1 (local refinement):
    chains are initialized from the amortized posterior, proposals follow a
    per-dimension Gaussian random walk scaled by std_i of the amortized
    posterior (times ``proposal_std``), proposals outside the task bounds are
    reflected back into the domain, and proposals are accepted according to
    the Metropolis--Hastings ratio built from the log prior and the log
    approximate likelihood (both evaluated in log space).
    
    Args:
        current_theta: (n_chains, d) Initial samples from the amortized posterior.
        x_obs_stats: (1, p) Target summary statistics.
        task: Task object exposing ``lower``/``upper`` bounds and, optionally,
            ``symmetry_dims`` (dimensions whose marginals are sign-ambiguous).
        log_prior_fn: Function (theta) -> log prior density.
        log_likelihood_fn: Function (theta, x_obs_stats) -> log approximate
            likelihood (numpy array, already in log space).
        n_chains: number of parallel chains.
        n_samples: number of posterior samples kept per chain.
        burn_in: number of burn-in steps.
        thin: thinning interval during the sampling phase.
        proposal_std: proposal scale multiplier applied to std_per_dim.

    Returns:
        Posterior samples with shape (n_chains * n_samples, d).
    """
    lower = np.asarray(getattr(task, "lower", None), dtype=np.float64)
    upper = np.asarray(getattr(task, "upper", None), dtype=np.float64)
    if lower is None or upper is None:
        raise ValueError("task must expose lower/upper bounds for refinement")
    bounded_mask = np.isfinite(lower) & np.isfinite(upper)
    current_theta = reflect_into_bounds_masked(np.asarray(current_theta, dtype=np.float64), lower, upper, bounded_mask)
    current_theta = prepare_theta_in_bounds(current_theta, lower, upper, context="MCMC init")
    
    # Dimensional-wise Proposal Step Size
    # θ_3 and θ_4 have symmetric bimodal marginals around 0: amortised draws
    # contain both signs, so a plain std across the mixture would be inflated
    # by the inter-mode spread and make the proposal scale too large.  We
    # therefore compute the std on |θ| for these dims (spread within a mode).
    # Note: abs() is applied only to the std copy; the MCMC chain keeps the
    # signed values, so the sampled posterior stays bimodal (no sign folding).
    symmetry_dims = getattr(task, "symmetry_dims", None)
    theta_for_std = np.asarray(current_theta, dtype=np.float64)
    if symmetry_dims is not None and len(symmetry_dims) > 0:
        theta_for_std = theta_for_std.copy()
        theta_for_std[:, symmetry_dims] = np.abs(theta_for_std[:, symmetry_dims])
    std_per_dim = np.std(theta_for_std, axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    proposal_scale = std_per_dim * proposal_std
    print(f"Proposal Scale (per dim): {proposal_scale}")
    
    # Initial log probabilities: log p(theta | x) = log p(theta) + log p(x | theta)
    current_log_prior = log_prior_fn(current_theta)
    current_log_like = np.asarray(log_likelihood_fn(current_theta, x_obs_stats), dtype=np.float64)
    current_log_prob = np.asarray(current_log_prior, dtype=np.float64) + current_log_like
    
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
        proposed_theta_raw, proposed_theta, reflected_mask = propose_mixed_random_walk(
            current_theta, proposal_scale, lower, upper, bounded_mask
        )
        total_reflected_all += np.sum(reflected_mask)
        proposed_theta = prepare_theta_in_bounds(proposed_theta, lower, upper, context="MCMC propose")
        
        # Proposed log prob
        proposed_log_prior = log_prior_fn(proposed_theta)
        proposed_log_like = np.asarray(log_likelihood_fn(proposed_theta, x_obs_stats), dtype=np.float64)
        proposed_log_prob = np.asarray(proposed_log_prior, dtype=np.float64) + proposed_log_like
        
        # Metropolis-Hastings acceptance in log space
        log_alpha = proposed_log_prob - current_log_prob
        log_alpha = np.where(np.isfinite(log_alpha), log_alpha, -np.inf)
        u = np.log(np.random.rand(n_chains))
        accept_mask = u < log_alpha
        total_accepted_all += np.sum(accept_mask)
        
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

def gaussian_log_likelihood(theta, x_obs_2d):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    x = np.asarray(x_obs_2d, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x_obs_2d shape (n, 2), got {x.shape}")
    n_obs = x.shape[0]
    m0 = theta[:, 0]
    m1 = theta[:, 1]
    s0_param = theta[:, 2]
    s1_param = theta[:, 3]
    r_param = theta[:, 4]
    s0_real = s0_param**2
    s1_real = s1_param**2
    rho = np.tanh(r_param)
    a = s0_real**2
    c = s1_real**2
    b = rho * s0_real * s1_real
    eps = 1e-8
    det = a * c - b**2
    det = np.maximum(det, eps)
    inv00 = c / det
    inv01 = -b / det
    inv11 = a / det
    dx0 = x[:, 0][np.newaxis, :] - m0[:, np.newaxis]
    dx1 = x[:, 1][np.newaxis, :] - m1[:, np.newaxis]
    q = inv00[:, np.newaxis] * dx0**2 + 2.0 * inv01[:, np.newaxis] * dx0 * dx1 + inv11[:, np.newaxis] * dx1**2
    q_sum = np.sum(q, axis=1)
    log_det = np.log(det)
    two_log2pi = 2.0 * np.log(2.0 * np.pi)
    common = two_log2pi + log_det
    log_likelihood = -0.5 * (n_obs * common + q_sum)
    return log_likelihood

def run_gaussian_posterior_mcmc(
    x_obs_2d,
    task,
    n_draws=2000,
    n_tune_chain=1000,
    chains=4,
    proposal_scale=0.3,
    return_acceptance_rate: bool = False,
):
    x = np.asarray(x_obs_2d, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x_obs_2d shape (n, 2), got {x.shape}")
    lower = np.asarray(getattr(task, "lower", None), dtype=np.float64)
    upper = np.asarray(getattr(task, "upper", None), dtype=np.float64)
    if lower is None or upper is None:
        raise ValueError("task must expose lower/upper bounds for true posterior mcmc")
    bounded_mask = np.isfinite(lower) & np.isfinite(upper)

    current_theta = np.asarray(task.sample_prior(chains), dtype=np.float64)
    current_theta = reflect_into_bounds_masked(current_theta, lower, upper, bounded_mask)
    current_theta = prepare_theta_in_bounds(current_theta, lower, upper, context="True posterior MCMC init")
    current_log_prior = np.asarray(task.log_prior(current_theta), dtype=np.float64)
    current_log_like = gaussian_log_likelihood(current_theta, x)
    current_log_post = current_log_prior + current_log_like
    samples = []
    total_accepted = 0
    total_steps = int(n_tune_chain) + int(n_draws)
    if total_steps <= 0:
        raise ValueError("Total number of MCMC steps must be positive.")
    print(f"Starting Gaussian posterior MCMC: chains={chains}, draws={n_draws}, tune={n_tune_chain}")
    for step in range(total_steps):
        proposal_scale_vec = np.full((task.d,), float(proposal_scale), dtype=np.float64)
        _, proposal, _ = propose_mixed_random_walk(
            current_theta, proposal_scale_vec, lower, upper, bounded_mask
        )
        proposal = prepare_theta_in_bounds(proposal, lower, upper, context="True posterior MCMC propose")
        proposed_log_prior = np.asarray(task.log_prior(proposal), dtype=np.float64)
        proposed_log_like = gaussian_log_likelihood(proposal, x)
        proposed_log_post = proposed_log_prior + proposed_log_like
        log_alpha = proposed_log_post - current_log_post
        log_alpha = np.where(np.isfinite(log_alpha), log_alpha, -np.inf)
        u = np.log(np.random.rand(chains))
        accept_mask = u < log_alpha
        current_theta[accept_mask] = proposal[accept_mask]
        current_log_post[accept_mask] = proposed_log_post[accept_mask]
        if step >= n_tune_chain:
            samples.append(current_theta.copy())
            total_accepted += np.sum(accept_mask)
        if (step + 1) % 50 == 0 or step == total_steps - 1:
            accepted_frac = np.mean(accept_mask)
            print(f"Step {step+1}/{total_steps} | Accepted fraction this step: {accepted_frac:.2%}")
    if not samples:
        samples.append(current_theta.copy())
    chain_samples = np.stack(samples, axis=1)
    flat_samples = chain_samples.reshape(-1, task.d)
    effective_draws = len(samples)
    acceptance_rate = total_accepted / (chains * max(effective_draws, 1))
    print(f"MCMC complete. Stored draws per chain: {effective_draws}, overall acceptance: {acceptance_rate:.2%}")
    if return_acceptance_rate:
        return flat_samples, float(acceptance_rate)
    return flat_samples

def compute_bandwidth_core(theta0_np, x_obs_tensor, task, stats_fn, n_samples, quantile_level, device):
    """
    Adaptive threshold of Algorithm 1: draw ``n_samples`` datasets from the
    amortized posterior at ``theta0_np``, compute the summaries S(X'), and set
    epsilon = ``quantile_level``-th quantile of ||S(X') - S(X*)||.
    """
    # Simulate Data from Theta0
    xn_0 = task.simulator(theta0_np, n_samples=task.n) 
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
    return quan1

def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, nsims, epsilon, device):
    """
    Approximate likelihood of Algorithm 1, evaluated in log space:
        log ell(theta) = log( (1/n_sims) * sum_{j=1}^{n_sims}
            exp( -||S(X_j) - S(X*)||^2 / (2 epsilon^2) ) ),
    where X_j ~ P_theta^(n) are n_sims i.i.d. simulated datasets at theta.
    """
    theta = np.asarray(theta)
    if theta.ndim != 2:
        raise ValueError(f"Expected theta with shape (batch, d), got {theta.shape}")

    batch_size = theta.shape[0]
    n_reps = max(int(nsims), 1)
    theta_rep = np.repeat(theta, n_reps, axis=0)
    sim_data = task.simulator(theta_rep, n_samples=task.n)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    
    with torch.no_grad():
        sim_stats = stats_fn(sim_data_tensor).detach().cpu().to(torch.float64)
        
    sim_stats = sim_stats.view(batch_size, n_reps, -1)
    target = x_obs_stats.detach().cpu().to(torch.float64).unsqueeze(0)
    
    diff = sim_stats - target
    dist_sq = torch.sum(diff**2, dim=-1)
    
    log_kernel = -dist_sq / (2.0 * float(epsilon) ** 2)
    log_lik = torch.logsumexp(log_kernel, dim=1) - np.log(n_reps)
    return log_lik.cpu().numpy()

# =============================================================================
# SMMD/MMD Implementation
# =============================================================================

def compute_bandwidth_smmd(model, x_obs, task, n_samples=20000, quantile_level=0.01, device="cpu"):
    print("Computing bandwidth for SMMD refinement...")
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1)
    
    with torch.no_grad():
        stats = model.T(x_obs_batch)
        Z = torch.randn(n_samples, 1, task.d, device=device)
        Theta0 = model.G(Z, stats).squeeze(1)
    Theta0_np = Theta0.cpu().numpy()
    
    return compute_bandwidth_core(Theta0_np, x_obs_tensor, task, model.T, n_samples, quantile_level, device)

def refine_posterior_smmd(model, x_obs, task, 
                         n_chains=2000, n_samples=1, burn_in=99, 
                         thin=1, nsims=1, epsilon=None, proposal_std=0.2, device="cpu",
                         quantile_level=0.01, bandwidth_n_samples=20000):
    print(f"Starting SMMD MCMC Refinement...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_smmd(model, x_obs, task,
                                         n_samples=int(bandwidth_n_samples),
                                         quantile_level=float(quantile_level),
                                         device=device)
        
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # 1. Stats and Initial Samples
    with torch.no_grad():
        x_obs_stats = model.T(x_obs_tensor)
        # Initial samples
        Z = torch.randn(n_chains, 1, task.d, device=device)
        current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()
        
    # 2. Define Likelihood Function
    def log_likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.T, nsims, epsilon, device)
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              task.log_prior, log_likelihood_fn, 
                              n_chains, n_samples, burn_in, thin, proposal_std)

# =============================================================================
# BayesFlow Implementation
# =============================================================================

def compute_bandwidth_bayesflow(model, x_obs, task, n_samples=20000, quantile_level=0.01, device="cpu"):
    print("Computing bandwidth for BayesFlow refinement...")
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # BayesFlow adapter needs CPU/Numpy
    x_obs_cpu = x_obs_tensor.cpu().numpy()
    
    # Sample from BayesFlow model
    # model.sample(conditions=...) returns (n_samples, batch, d) or (batch, n_samples, d)
    
    # Update for Keras 3 with explicit adapter (requires dict)
    conditions_dict = {"summary_variables": x_obs_cpu}
    Theta0_np = model.sample(conditions=conditions_dict, num_samples=n_samples)
    
    # Handle Dict output (BayesFlow/Keras3)
    if isinstance(Theta0_np, dict):
        Theta0_np = Theta0_np["inference_variables"]
    
    # Theta0_np shape: (1, n_samples, d) (since batch_size=1)
    # Ensure numpy array
    if isinstance(Theta0_np, torch.Tensor):
        Theta0_np = Theta0_np.detach().cpu().numpy()
    elif not isinstance(Theta0_np, np.ndarray):
         Theta0_np = np.asarray(Theta0_np)
         
    Theta0_np = Theta0_np.reshape(n_samples, -1)
        
    return compute_bandwidth_core(Theta0_np, x_obs_tensor, task, model.summary_network, n_samples, quantile_level, device)

def refine_posterior_bayesflow(model, x_obs, task, 
                              n_chains=2000, n_samples=1, burn_in=99, 
                              thin=1, nsims=1, epsilon=None, proposal_std=0.2, device="cpu",
                              quantile_level=0.01, bandwidth_n_samples=20000):
    print(f"Starting BayesFlow MCMC Refinement...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_bayesflow(model, x_obs, task,
                                              n_samples=int(bandwidth_n_samples),
                                              quantile_level=float(quantile_level),
                                              device=device)
        
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # 1. Stats and Initial Samples
    with torch.no_grad():
        x_obs_stats = model.summary_network(x_obs_tensor)
        
        # BayesFlow adapter needs CPU/Numpy
        x_obs_cpu = x_obs_tensor.cpu().numpy()
        conditions_dict = {"summary_variables": x_obs_cpu}
        current_theta = model.sample(conditions=conditions_dict, num_samples=n_chains)
        
        # Handle Dict output (BayesFlow/Keras3)
        if isinstance(current_theta, dict):
            current_theta = current_theta["inference_variables"]
            
        # Ensure numpy array
        if isinstance(current_theta, torch.Tensor):
            current_theta = current_theta.detach().cpu().numpy()
        elif not isinstance(current_theta, np.ndarray):
             current_theta = np.asarray(current_theta)
             
        current_theta = current_theta.reshape(n_chains, -1)
            
    # 2. Define Likelihood Function
    def log_likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.summary_network, nsims, epsilon, device)
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              task.log_prior, log_likelihood_fn, 
                              n_chains, n_samples, burn_in, thin, proposal_std)

# Backwards compatibility / Dispatcher (optional)
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
# MMD Metric (Evaluation)
# =============================================================================

def compute_mmd_metric(samples_approx, samples_true):
    """
    Compute the squared Maximum Mean Discrepancy (MMD^2) between approximate and true posterior samples.
    Kernel: Gaussian RBF kernel k(x, y) = exp(-||x-y||^2 / h^2)
    Bandwidth h: Median of pairwise L2 distances of true posterior samples.
    We report the squared MMD (V-statistic form),
    MMD^2 = mean(K_XX) + mean(K_YY) - 2*mean(K_XY);
    the paper tables report MMD^2 x 10^2.
    
    Args:
        samples_approx: (n_samples_a, d) numpy array
        samples_true: (n_samples_b, d) numpy array
    Returns:
        mmd_value: Scalar squared MMD value (MMD^2)
    """
    # Ensure numpy arrays
    if isinstance(samples_approx, torch.Tensor):
        samples_approx = samples_approx.detach().cpu().numpy()
    if isinstance(samples_true, torch.Tensor):
        samples_true = samples_true.detach().cpu().numpy()
        
    X = np.asarray(samples_approx, dtype=float)
    Y = np.asarray(samples_true, dtype=float)
    
    ny = Y.shape[0]
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"Expected X and Y to be 2D with same parameter dimension, got X={X.shape}, Y={Y.shape}")
    
    # 1. Compute Bandwidth h from True Samples (Y)
    Y_for_bw = np.asarray(Y, dtype=float)
    if Y_for_bw.ndim == 2 and Y_for_bw.shape[1] >= 4:
        Y_for_bw = Y_for_bw.copy()
        Y_for_bw[:, 2] = np.abs(Y_for_bw[:, 2])
        Y_for_bw[:, 3] = np.abs(Y_for_bw[:, 3])

    if ny < 2:
        h_sq = 1.0
    else:
        Y_t = torch.from_numpy(np.asarray(Y_for_bw, dtype=np.float32))
        Y_sq = torch.sum(Y_t * Y_t, dim=1, keepdim=True)
        D_YY_sq = Y_sq + Y_sq.T - 2.0 * (Y_t @ Y_t.T)
        D_YY_sq = torch.clamp(D_YY_sq, min=0.0)
        triu = torch.triu_indices(ny, ny, offset=1)
        pairwise_dists = torch.sqrt(D_YY_sq[triu[0], triu[1]])
        h = torch.median(pairwise_dists).item()
        h_sq = float(h * h)

    if (not np.isfinite(h_sq)) or h_sq < 1e-12:
        h_sq = 1.0
        
    # 2. Compute MMD^2
    # k(x, y) = exp(- ||x-y||^2 / h^2)
    
    def mean_rbf_kernel(A, B, h_sq, block_size=1024):
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        h_sq = np.float32(h_sq)

        total = 0.0
        count = 0
        for i0 in range(0, A.shape[0], block_size):
            i1 = min(i0 + block_size, A.shape[0])
            A_blk = A[i0:i1]
            A_sq = np.sum(A_blk * A_blk, axis=1, keepdims=True)
            for j0 in range(0, B.shape[0], block_size):
                j1 = min(j0 + block_size, B.shape[0])
                B_blk = B[j0:j1]
                B_sq = np.sum(B_blk * B_blk, axis=1, keepdims=True)
                D = A_sq + B_sq.T - 2.0 * (A_blk @ B_blk.T)
                np.maximum(D, 0.0, out=D)
                total += float(np.exp(-D / h_sq).sum())
                count += D.size
        return total / max(count, 1)

    mean_k_xx = mean_rbf_kernel(X, X, h_sq)
    mean_k_yy = mean_rbf_kernel(Y, Y, h_sq)
    mean_k_xy = mean_rbf_kernel(X, Y, h_sq)
    
    # Unbiased MMD^2 estimator usually excludes diagonal for XX and YY, 
    # but standard V-statistic includes them. 
    # Let's use the standard form: mean(K_XX) + mean(K_YY) - 2*mean(K_XY)
    
    mmd_sq = mean_k_xx + mean_k_yy - 2 * mean_k_xy
    
    # Numerical noise can make mmd_sq slightly negative; clip at zero.
    # We report the squared MMD (MMD^2), consistent with the paper tables.
    mmd_value = np.maximum(mmd_sq, 0)
    
    return mmd_value


def compute_swd_metric(
    samples_approx,
    samples_true,
    n_projections: int = 1000,
    max_samples: int = 1000,
    seed: int = 0,
):
    if isinstance(samples_approx, torch.Tensor):
        samples_approx = samples_approx.detach().cpu().numpy()
    if isinstance(samples_true, torch.Tensor):
        samples_true = samples_true.detach().cpu().numpy()

    X = np.asarray(samples_approx, dtype=np.float32)
    Y = np.asarray(samples_true, dtype=np.float32)

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Expected X and Y to be 2D with same parameter dimension, got X={X.shape}, Y={Y.shape}"
        )

    n_x = X.shape[0]
    n_y = Y.shape[0]
    if n_x < 2 or n_y < 2:
        return float("nan")

    rng = np.random.RandomState(int(seed))
    n_use = int(min(max_samples, n_x, n_y))
    idx_x = rng.choice(n_x, size=n_use, replace=False)
    idx_y = rng.choice(n_y, size=n_use, replace=False)
    X_use = X[idx_x]
    Y_use = Y[idx_y]

    import ot

    swd = ot.sliced_wasserstein_distance(
        X_use,
        Y_use,
        n_projections=int(n_projections),
        seed=int(seed),
    )
    return float(swd)
