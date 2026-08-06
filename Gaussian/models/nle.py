import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import NLE_A
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.samplers.mcmc.slice_numpy import SliceSamplerVectorized
from sbi.neural_nets import likelihood_nn
from sbi.utils import BoxUniform


def _resolve_sbi_device(requested_device: str) -> torch.device:
    requested = str(requested_device).strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    # MPS currently stays on CPU for sbi to avoid prior/device mismatches.
    return torch.device("cpu")


def _build_singleton_pairs(
    theta: np.ndarray,
    x: np.ndarray,
    observations_per_theta: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N, n_obs, d_x), got {x.shape}")
    if theta.ndim != 2:
        raise ValueError(f"Expected theta with shape (N, d_theta), got {theta.shape}")
    if x.shape[0] != theta.shape[0]:
        raise ValueError(
            f"Mismatched N: theta has {theta.shape[0]} rows but x has {x.shape[0]} rows."
        )

    n_sims, n_obs, _ = x.shape
    k = int(observations_per_theta)
    if k <= 0:
        k = int(n_obs)

    if k >= n_obs:
        x_pairs = x.reshape(n_sims * n_obs, -1)
        theta_pairs = np.repeat(theta, repeats=n_obs, axis=0)
        return theta_pairs.astype(np.float32), x_pairs.astype(np.float32)

    obs_idx = rng.randint(0, n_obs, size=(n_sims, k))
    rows = np.arange(n_sims)[:, None]
    x_pairs = x[rows, obs_idx].reshape(n_sims * k, -1)
    theta_pairs = np.repeat(theta, repeats=k, axis=0)
    return theta_pairs.astype(np.float32), x_pairs.astype(np.float32)


@dataclass
class NLESBITrainSummary:
    training_time_sec: float
    sbi_summary: Dict[str, Any]


class NLESBIOfficialFactorized(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        prior_lower: np.ndarray,
        prior_upper: np.ndarray,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        z_score_theta: str = "independent",
        z_score_x: str = "independent",
        observations_per_theta: int = 10,
        mcmc_method: str = "slice_np_vectorized",
        mcmc_chains: int = 64,
        mcmc_chain_batch_size: Optional[int] = None,
        mcmc_warmup_steps: int = 200,
        mcmc_thin: int = 1,
        slice_width_mode: str = "constant",
        slice_init_width: float = 0.01,
        slice_reference_n: float = 50.0,
        seed: int = 0,
        device: str = "auto",
        prior_type: str = "uniform",
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_features = int(hidden_features)
        self.num_transforms = int(num_transforms)
        self.num_blocks = int(num_blocks)
        self.z_score_theta = str(z_score_theta)
        self.z_score_x = str(z_score_x)
        self.observations_per_theta = int(observations_per_theta)
        self.mcmc_method = str(mcmc_method)
        self.mcmc_chains = int(mcmc_chains)
        self.mcmc_chain_batch_size = (
            None if mcmc_chain_batch_size is None else int(mcmc_chain_batch_size)
        )
        self.mcmc_warmup_steps = int(mcmc_warmup_steps)
        self.mcmc_thin = int(mcmc_thin)
        self.slice_width_mode = str(slice_width_mode).lower()
        self.slice_init_width = float(slice_init_width)
        self.slice_reference_n = float(slice_reference_n)
        self.seed = int(seed)
        self.prior_type = str(prior_type).lower()
        self.sbi_device = _resolve_sbi_device(device)
        self.device = str(self.sbi_device)
        self.posterior_device = (
            torch.device("cpu")
            if self.mcmc_method == "slice_np_vectorized"
            else self.sbi_device
        )

        low = torch.as_tensor(prior_lower, dtype=torch.float32, device=self.sbi_device)
        high = torch.as_tensor(prior_upper, dtype=torch.float32, device=self.sbi_device)
        self.prior = BoxUniform(low=low, high=high)
        low_cpu = torch.as_tensor(prior_lower, dtype=torch.float32)
        high_cpu = torch.as_tensor(prior_upper, dtype=torch.float32)
        self._cpu_prior = BoxUniform(low=low_cpu, high=high_cpu)

        self._density_builder = likelihood_nn(
            model="maf",
            z_score_theta=self.z_score_theta,
            z_score_x=self.z_score_x,
            hidden_features=self.hidden_features,
            num_transforms=self.num_transforms,
            embedding_net=nn.Identity(),
            num_blocks=self.num_blocks,
            use_batch_norm=False,
        )

        self.inference = NLE_A(
            prior=self.prior,
            density_estimator=self._density_builder,
            device=self.device,
            show_progress_bars=True,
        )

        self.likelihood_estimator = None
        self._posterior = None

    def _resolve_slice_init_width(self, n_obs: int) -> float:
        width = float(self.slice_init_width)
        if self.slice_width_mode == "sqrt_n":
            width = width * math.sqrt(self.slice_reference_n / float(n_obs))
        return max(width, 1e-6)

    def _sample_posterior_slice_vectorized(
        self,
        potential_fn,
        x_o: torch.Tensor,
        n_samples: int,
    ) -> np.ndarray:
        n_obs = int(x_o.shape[0])
        init_width = self._resolve_slice_init_width(n_obs)
        total_chains = max(1, int(self.mcmc_chains))
        draws_per_chain = max(1, int(math.ceil(float(n_samples) / float(total_chains))))
        chain_batch_size = total_chains
        if self.mcmc_chain_batch_size is not None and self.mcmc_chain_batch_size > 0:
            chain_batch_size = min(total_chains, int(self.mcmc_chain_batch_size))

        def log_prob_np(theta_np: np.ndarray) -> np.ndarray:
            theta_t = torch.as_tensor(theta_np, dtype=torch.float32)
            values = potential_fn(theta_t, x_o)
            return values.detach().cpu().numpy().astype(np.float64)

        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            collected = []
            chains_remaining = total_chains
            while chains_remaining > 0:
                curr_chains = min(chain_batch_size, chains_remaining)
                init_params = (
                    self._cpu_prior.sample((curr_chains,))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                sampler = SliceSamplerVectorized(
                    log_prob_fn=log_prob_np,
                    init_params=init_params,
                    num_chains=curr_chains,
                    thin=max(1, int(self.mcmc_thin)),
                    tuning=max(0, int(self.mcmc_warmup_steps)),
                    verbose=True,
                    init_width=init_width,
                    num_workers=1,
                )
                chain_samples = sampler.run(draws_per_chain)
                collected.append(
                    np.asarray(chain_samples, dtype=np.float32).reshape(-1, self.theta_dim)
                )
                chains_remaining -= curr_chains
        finally:
            np.random.set_state(state)

        samples = np.concatenate(collected, axis=0)
        return samples[: int(n_samples)]

    def _log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        theta_b = theta if theta.ndim == 2 else theta.unsqueeze(0)
        if self.prior_type != "conditional":
            theta_prior = theta_b.to(self.prior.base_dist.low.device)
            return self.prior.log_prob(theta_prior).to(theta.device)

        lower = self.prior.base_dist.low.to(theta_b.device)
        upper = self.prior.base_dist.high.to(theta_b.device)
        in_bounds = ((theta_b >= lower) & (theta_b <= upper)).all(dim=1)
        log_prob = torch.full(
            (theta_b.shape[0],),
            fill_value=-torch.inf,
            dtype=theta_b.dtype,
            device=theta_b.device,
        )
        if torch.any(in_bounds):
            theta_in = theta_b[in_bounds]
            sigma = torch.tensor(0.1, dtype=theta_b.dtype, device=theta_b.device)
            mu = theta_in[:, 0].pow(2)
            t1 = theta_in[:, 1]
            a = (t1 - mu) / sigma
            alpha = (lower[1] - mu) / sigma
            beta = (upper[1] - mu) / sigma
            sqrt_two = torch.tensor(math.sqrt(2.0), dtype=theta_b.dtype, device=theta_b.device)
            z = 0.5 * (1.0 + torch.erf(beta / sqrt_two)) - 0.5 * (1.0 + torch.erf(alpha / sqrt_two))
            z = torch.clamp(z, min=1e-300)
            log_prob[in_bounds] = (
                -0.5 * a.pow(2)
                - torch.log(sigma)
                - 0.5 * math.log(2.0 * math.pi)
                - torch.log(z)
            )
        return log_prob if theta.ndim == 2 else log_prob[0]

    def fit(
        self,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        max_num_epochs: int,
        learning_rate: float,
        training_batch_size: int,
        validation_fraction: float,
        stop_after_epochs: int,
    ) -> NLESBITrainSummary:
        rng = np.random.RandomState(self.seed)
        theta_pairs, x_pairs = _build_singleton_pairs(
            theta_train,
            x_train,
            observations_per_theta=self.observations_per_theta,
            rng=rng,
        )

        theta_t = torch.from_numpy(theta_pairs).float().to(self.sbi_device)
        x_t = torch.from_numpy(x_pairs).float().to(self.sbi_device)

        t0 = time.time()
        self.inference.append_simulations(theta_t, x_t)
        with torch.enable_grad():
            self.likelihood_estimator = self.inference.train(
                training_batch_size=int(training_batch_size),
                learning_rate=float(learning_rate),
                validation_fraction=float(validation_fraction),
                stop_after_epochs=int(stop_after_epochs),
                max_num_epochs=int(max_num_epochs),
                clip_max_norm=5.0,
                show_train_summary=False,
            )
        t1 = time.time()

        return NLESBITrainSummary(
            training_time_sec=float(t1 - t0),
            sbi_summary=dict(getattr(self.inference, "summary", {}) or {}),
        )

    @torch.no_grad()
    def sample_posterior(self, x_obs: np.ndarray, n_samples: int) -> np.ndarray:
        if self.likelihood_estimator is None:
            raise RuntimeError("Call fit() before sample_posterior().")

        x_o = torch.as_tensor(x_obs, dtype=torch.float32)
        if x_o.ndim == 1:
            if x_o.shape[0] != self.x_dim:
                raise ValueError(f"Expected x_obs with last dim {self.x_dim}, got {x_o.shape}")
            x_o = x_o.unsqueeze(0)
        elif x_o.ndim == 2:
            if x_o.shape[1] != self.x_dim:
                raise ValueError(f"Expected x_obs with last dim {self.x_dim}, got {x_o.shape}")
        else:
            raise ValueError(f"Expected x_obs with shape (n_obs, d_x) or (d_x,), got {x_o.shape}")

        def potential_fn(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 1:
                x_eval = x.unsqueeze(0)
            elif x.ndim == 2:
                x_eval = x
            else:
                raise ValueError(f"Unexpected x shape in potential: {x.shape}")

            theta_b = theta
            if theta_b.ndim == 1:
                theta_b = theta_b.unsqueeze(0)

            batch = theta_b.shape[0]
            n_obs = x_eval.shape[0]
            eval_device = next(self.likelihood_estimator.parameters()).device

            x_rep = x_eval.unsqueeze(0).expand(batch, n_obs, self.x_dim).reshape(batch * n_obs, self.x_dim)
            theta_rep = theta_b.unsqueeze(1).expand(batch, n_obs, self.theta_dim).reshape(batch * n_obs, self.theta_dim)
            x_rep = x_rep.to(eval_device)
            theta_rep = theta_rep.to(eval_device)
            log_like_singletons = self.likelihood_estimator.log_prob(
                x_rep.unsqueeze(0), theta_rep
            )[0]
            log_like = log_like_singletons.reshape(batch, n_obs).sum(dim=1)
            log_prior = self._log_prior(theta_b).to(log_like.device)
            return (log_prior.to(log_like.device) + log_like).to(theta.device)

        if self.mcmc_method == "slice_np_vectorized":
            return self._sample_posterior_slice_vectorized(
                potential_fn=potential_fn,
                x_o=x_o,
                n_samples=int(n_samples),
            )

        if self._posterior is None:
            proposal_prior = (
                self._cpu_prior if self.posterior_device.type == "cpu" else self.prior
            )
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                proposal=proposal_prior,
                theta_transform=None,
                method=self.mcmc_method,
                thin=self.mcmc_thin,
                warmup_steps=self.mcmc_warmup_steps,
                num_chains=max(1, int(self.mcmc_chains)),
                init_strategy="proposal",
                device=str(self.posterior_device),
            )

        self._posterior.set_default_x(x_o)
        samples = self._posterior.sample((int(n_samples),), x=x_o, show_progress_bars=True)
        return samples.detach().cpu().numpy().astype(np.float32)
