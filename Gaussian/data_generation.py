"""
Gaussian Data Generation and Reference Posterior (MCMC).
Refactored to include GaussianTask class.
"""

import numpy as np
import scipy.stats

# ============================================================================
# 1. Configuration
# ============================================================================

# Data parameters
N = 12800    # Size of the training dataset
n = 50       # Number of samples per observation (sample size)
d = 5        # Dimension of parameter theta (m0, m1, s0, s1, r)
d_x = 3      # Dimension of x (after stereographic projection)
p = 10       # Dimension of summary statistics

# Ground Truth Parameters
TRUE_PARAMS = np.array([1.0, 1.0, -1.0, -0.9, 0.6]) 

# ============================================================================
# 2. Gaussian Task Class (Prior, Simulator, Likelihood)
# ============================================================================

class GaussianTask:
    def __init__(self, n=n, prior_type='conditional'):
        """
        Initialize Gaussian Task.
        n: Number of samples per observation.
        prior_type: 'uniform' (all independent U[-3, 3])
                 or 'conditional' (theta[1] | theta[0] ~ TruncNormal(theta[0]**2, 0.1^2; [-3, 3]))
        """
        self.prior_type = prior_type
        self.d = d
        self.n = n
        self.d_x = d_x
        self.lower = np.array([-3.0] * d, dtype=np.float64)
        self.upper = np.array([+3.0] * d, dtype=np.float64)
        # θ_3 and θ_4 (0-indexed: 2, 3) each have marginal posteriors with two
        # symmetric modes around 0 (e.g. +1.0 and -1.0).  The sign is therefore
        # arbitrary, and |θ| is the quantity of interest for MCMC proposal scale.
        # MCMC proposal std uses abs() for these dims to avoid variance inflation
        # from the bimodal sign ambiguity in the amortised posterior.
        self.symmetry_dims = [2, 3]
        
        print(f"Initialized GaussianTask with prior_type='{self.prior_type}'")

    def sample_prior(self, batch_size):
        """
        Sample parameters from the prior.
        """
        theta = np.empty((batch_size, self.d), dtype=np.float64)
        theta[:, 0] = np.random.uniform(-3.0, 3.0, size=batch_size)
        theta[:, 2] = np.random.uniform(-3.0, 3.0, size=batch_size)
        theta[:, 3] = np.random.uniform(-3.0, 3.0, size=batch_size)
        theta[:, 4] = np.random.uniform(-3.0, 3.0, size=batch_size)

        if self.prior_type == 'conditional':
            mu = theta[:, 0] ** 2
            sigma = 0.1
            alpha = (self.lower[1] - mu) / sigma
            beta = (self.upper[1] - mu) / sigma
            theta[:, 1] = scipy.stats.truncnorm.rvs(
                a=alpha,
                b=beta,
                loc=mu,
                scale=sigma,
                size=batch_size,
            )
        else:
            theta[:, 1] = np.random.uniform(-3.0, 3.0, size=batch_size)

        return theta.astype(np.float32)

    def log_prior(self, theta):
        """
        Calculate log prior probability.
        theta: (d,) or (batch, d)
        Returns: scalar or (batch,)
        """
        theta = np.asarray(theta)
        is_batch = theta.ndim > 1
        if not is_batch:
            theta = theta[np.newaxis, :]
            
        batch_size = theta.shape[0]
        log_probs = np.zeros(batch_size, dtype=np.float32)
        
        out_of_bounds = np.any((theta < self.lower) | (theta > self.upper), axis=1)
        log_probs[out_of_bounds] = -np.inf
        
        if self.prior_type == 'conditional':
            # log p(theta) = log p(theta_1 | theta_0) under a truncated normal on [-3, 3],
            # plus constants from the bounded uniform dims.
            mask = ~out_of_bounds
            if np.any(mask):
                t0 = theta[mask, 0]
                t1 = theta[mask, 1]
                sigma = 0.1
                mu = t0**2
                a = (t1 - mu) / sigma
                alpha = (self.lower[1] - mu) / sigma
                beta = (self.upper[1] - mu) / sigma
                z = scipy.stats.norm.cdf(beta) - scipy.stats.norm.cdf(alpha)
                z = np.clip(z, 1e-300, None)
                log_probs[mask] += (
                    -0.5 * a**2
                    - np.log(sigma)
                    - 0.5 * np.log(2.0 * np.pi)
                    - np.log(z)
                )
        
        if not is_batch:
            return log_probs[0]
        return log_probs

    def simulator(self, theta, n_samples=None, rng=np.random):
        """
        Simulate data from Gaussian model with stereo projection.
        theta: (batch_size, 5) or (5,)
        Returns: (batch_size, n_samples, 3)
        """
        if n_samples is None:
            n_samples = self.n

        theta = np.asarray(theta)
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
            
        # Get raw 2D data
        xs_2d = self.simulator_2d(theta, n_samples, rng)
        
        # Stereo Projection
        xs_proj = self.stereo_proj(xs_2d)
        
        return xs_proj

    def simulator_2d(self, theta, n_samples=None, rng=np.random):
        """
        Internal simulator for 2D Gaussian data (before projection).
        """
        if n_samples is None:
            n_samples = self.n

        theta = np.asarray(theta)
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
        
        batch_size = theta.shape[0]
        
        m0 = theta[:, 0:1]
        m1 = theta[:, 1:2]
        s0 = theta[:, 2:3] ** 2
        s1 = theta[:, 3:4] ** 2
        r = np.tanh(theta[:, 4:5])
        
        us = rng.randn(batch_size, n_samples, 2)
        xs = np.empty_like(us)
        
        xs[..., 0] = s0 * us[..., 0] + m0
        term_r = r * us[..., 0] + np.sqrt(1.0 - r**2) * us[..., 1]
        xs[..., 1] = s1 * term_r + m1
        
        return xs

    def stereo_proj(self, A):
        """
        Spherical (stereographic) projection transform.
        A: (batch, n, 2)
        Returns: (batch, n, 3)
        """
        X_comp = A[..., 0]
        Y_comp = A[..., 1]
        denom = 1 + X_comp**2 + Y_comp**2
        
        new_X_comp = 2 * X_comp / denom
        new_Y_comp = 2 * Y_comp / denom
        Z_comp = (-1 + X_comp**2 + Y_comp**2) / denom
        
        result = np.stack([new_X_comp, new_Y_comp, Z_comp], axis=-1)
        return result

    def get_ground_truth(self):
        """
        Returns ground truth parameters and observations.
        Returns: (theta_true, x_obs_3d)
        """
        theta_true = TRUE_PARAMS
        theta_batch = theta_true[np.newaxis, :]
        
        # 3D Projected Data (for SMMD/BayesFlow)
        # simulator returns 3D data
        xs_3d = self.simulator(theta_batch, n_samples=self.n)[0]
        
        return theta_true, xs_3d

# ============================================================================
# 3. Helpers (Dataset Generation)
# ============================================================================

def generate_dataset(task=None, n_sims=N, n_obs=None):
    """
    Generates the Reference Table (Theta, X) for training.
    """
    if task is None:
        task = GaussianTask(prior_type='conditional')
    
    if n_obs is None:
        n_obs = task.n
        
    print(f"Generating dataset (N={n_sims})...")
    
    # 1. Sample Theta
    thetas = task.sample_prior(n_sims) # (N, 5)
    
    # 2. Simulate X
    xs = task.simulator(thetas, n_samples=n_obs).astype(np.float32) # (N, n, 3)
    
    return thetas, xs

if __name__ == "__main__":
    # Test
    task = GaussianTask(prior_type='conditional')
    t, x = generate_dataset(task, n_sims=10)
    print("Theta shape:", t.shape)
    
    true_p, obs_3d = task.get_ground_truth()
    print("Ground truth shapes:", true_p.shape, obs_3d.shape)
