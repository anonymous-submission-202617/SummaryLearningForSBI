"""Time-Diffusion Neural Posterior Score Estimation (TD-NPSE) model for the
Gaussian task."""

import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def prec_matrix_backward(t, dist_cov, model):
    alpha_t = model.alpha(t)
    sigma_t = model.sigma(t)
    eye = torch.eye(dist_cov.shape[-1], dtype=dist_cov.dtype, device=dist_cov.device)
    return torch.linalg.inv(dist_cov) + (alpha_t / sigma_t.pow(2)) * eye


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, num_frequencies=3):
        super().__init__()
        freq = torch.arange(1, num_frequencies + 1, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", freq)

    @property
    def output_dim(self):
        return 2 * int(self.frequencies.shape[0])

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        angles = t * self.frequencies.unsqueeze(0)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class TDNPSEScoreNet(nn.Module):
    def __init__(
        self,
        theta_dim,
        x_dim,
        hidden_dim=256,
        num_layers=3,
        time_frequencies=3,
        hidden_features=None,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(num_frequencies=time_frequencies)
        if hidden_features is None:
            hidden_features = [hidden_dim] * num_layers
        hidden_features = list(hidden_features)

        input_dim = theta_dim + x_dim + self.time_embedding.output_dim
        layers = []
        prev_dim = input_dim
        for width in hidden_features:
            layers.append(nn.Linear(prev_dim, int(width)))
            layers.append(nn.SiLU())
            prev_dim = int(width)
        layers.append(nn.Linear(prev_dim, theta_dim))
        self.net = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, theta_t, x, t):
        t_embed = self.time_embedding(t)
        return self.net(torch.cat([theta_t, x, t_embed], dim=-1))


class TDNPSE_Model(nn.Module):
    def __init__(
        self,
        theta_dim=5,
        x_dim=3,
        hidden_dim=256,
        num_layers=3,
        hidden_features=None,
        time_frequencies=3,
        num_diffusion_steps=100,
        beta_start=1e-4,
        beta_end=1e-1,
        ddim_eta=1.0,
        cov_reg=1e-3,
        cov_samples=256,
        cov_steps=100,
        sample_steps=1000,
        cov_mode="GAUSS",
        prior_lower=None,
        prior_upper=None,
        theta_mean=None,
        theta_std=None,
        x_mean=None,
        x_std=None,
        uniform_prior_scale=1.0,
        clip_denoised=True,
        prior_type="uniform",
        symmetry_sign_flip_dims=None,
        symmetry_mode_sampling_dims=None,
        sampler_type="ddim",
        tall_score_mode="gauss_refine",
        covariance_scale=1.0,
        covariance_symmetry_dims=None,
        prior_score_clip=50.0,
        tall_score_clip=100.0,
    ):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.ddim_eta = float(ddim_eta)
        self.cov_reg = float(cov_reg)
        self.cov_samples = int(cov_samples)
        self.cov_steps = int(cov_steps)
        self.sample_steps = int(sample_steps)
        self.cov_mode = str(cov_mode).upper()
        self.clip_denoised = bool(clip_denoised)
        self.uniform_prior_scale = float(uniform_prior_scale)
        self.prior_type = str(prior_type).lower()
        self.sampler_type = str(sampler_type).lower()
        self.tall_score_mode = str(tall_score_mode).lower()
        self.covariance_scale = float(covariance_scale)
        self.prior_score_clip = float(prior_score_clip)
        self.tall_score_clip = float(tall_score_clip)
        self.eps_floor = math.exp(-16.0)
        self._conditional_prior_warning_emitted = False
        if self.sampler_type not in {"ddim", "euler"}:
            raise ValueError(
                f"Unknown sampler_type='{self.sampler_type}'. Expected 'ddim' or 'euler'."
            )

        if prior_lower is None:
            prior_lower = -3.0 * np.ones(theta_dim, dtype=np.float32)
        if prior_upper is None:
            prior_upper = 3.0 * np.ones(theta_dim, dtype=np.float32)
        if theta_mean is None:
            theta_mean = np.zeros(theta_dim, dtype=np.float32)
        if theta_std is None:
            theta_std = np.ones(theta_dim, dtype=np.float32)
        if x_mean is None:
            x_mean = np.zeros(x_dim, dtype=np.float32)
        if x_std is None:
            x_std = np.ones(x_dim, dtype=np.float32)

        theta_std = np.asarray(theta_std, dtype=np.float32)
        theta_std = np.where(theta_std < 1e-6, 1.0, theta_std)
        x_std = np.asarray(x_std, dtype=np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)

        self.register_buffer("prior_lower", torch.as_tensor(prior_lower, dtype=torch.float32))
        self.register_buffer("prior_upper", torch.as_tensor(prior_upper, dtype=torch.float32))
        self.register_buffer("theta_mean", torch.as_tensor(theta_mean, dtype=torch.float32))
        self.register_buffer("theta_std", torch.as_tensor(theta_std, dtype=torch.float32))
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32))
        if symmetry_sign_flip_dims is None:
            symmetry_sign_flip_dims = []
        self.symmetry_sign_flip_dims = tuple(
            sorted(
                {
                    int(idx)
                    for idx in symmetry_sign_flip_dims
                    if 0 <= int(idx) < self.theta_dim
                }
            )
        )
        if symmetry_mode_sampling_dims is None:
            symmetry_mode_sampling_dims = []
        self.symmetry_mode_sampling_dims = tuple(
            sorted(
                {
                    int(idx)
                    for idx in symmetry_mode_sampling_dims
                    if 0 <= int(idx) < self.theta_dim
                }
            )
        )
        if covariance_symmetry_dims is None:
            covariance_symmetry_dims = []
        self.covariance_symmetry_dims = tuple(
            sorted(
                {
                    int(idx)
                    for idx in covariance_symmetry_dims
                    if 0 <= int(idx) < self.theta_dim
                }
            )
        )

        prior_lower_norm = ((self.prior_lower - self.theta_mean) / self.theta_std) * self.uniform_prior_scale
        prior_upper_norm = ((self.prior_upper - self.theta_mean) / self.theta_std) * self.uniform_prior_scale
        self.register_buffer("prior_lower_norm", prior_lower_norm)
        self.register_buffer("prior_upper_norm", prior_upper_norm)

        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.score_net = TDNPSEScoreNet(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_frequencies=time_frequencies,
            hidden_features=hidden_features,
        )

        if self.prior_type == "conditional":
            warnings.warn(
                "TD-NPSE tall-data composition is only theoretically supported when the "
                "diffused prior score grad log p_t^lambda(theta) is available in closed form "
                "(for this project: uniform prior). With prior_type='conditional', the code "
                "below can only use a heuristic prior term and should not be trusted for n_obs > 1. "
                "If you really need a conditional prior, train with classifier-free guidance (CFG) "
                "to learn a prior score surrogate instead; however, the original paper reports "
                "poor practical performance, so this path has limited reference value.",
                stacklevel=2,
            )

    def forward(self, theta_t, x, t):
        t = self._normalize_time_input(t, batch_size=theta_t.shape[0], device=theta_t.device)
        return self.score_net(theta_t, x, t)

    def _normalize_time_input(self, t, batch_size, device):
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32, device=device)
        else:
            t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(batch_size)
        elif t.ndim == 1 and t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size)
        return t

    def beta(self, t):
        return 32.0 * t

    def alpha(self, t):
        t = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=self.prior_lower.device if not torch.is_tensor(t) else t.device,
        )
        return torch.exp(-16.0 * t.pow(2))

    def sigma(self, t):
        alpha_t = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - alpha_t + self.eps_floor, min=1e-8))

    def score(self, theta_t, x, t):
        t = self._normalize_time_input(t, batch_size=theta_t.shape[0], device=theta_t.device)
        eps_pred = self(theta_t, x, t)
        return -eps_pred / self.sigma(t).unsqueeze(-1)

    def normalize_theta(self, theta):
        return (theta - self.theta_mean) / self.theta_std

    def unnormalize_theta(self, theta_norm):
        return theta_norm * self.theta_std + self.theta_mean

    def normalize_x(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        return torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_sign_flip_symmetry(self, theta_samples):
        if not self.symmetry_sign_flip_dims:
            return theta_samples
        if theta_samples.ndim != 2 or theta_samples.shape[0] == 0:
            return theta_samples

        dims = list(self.symmetry_sign_flip_dims)
        theta_sym = theta_samples.clone()
        theta_sym[:, dims] = theta_sym[:, dims].abs()

        n_patterns = 1 << len(dims)
        sign_patterns = torch.ones(
            n_patterns,
            len(dims),
            dtype=theta_sym.dtype,
            device=theta_sym.device,
        )
        for pattern_idx in range(n_patterns):
            for bit_idx in range(len(dims)):
                if (pattern_idx >> bit_idx) & 1:
                    sign_patterns[pattern_idx, bit_idx] = -1.0

        pattern_ids = torch.arange(theta_sym.shape[0], device=theta_sym.device) % n_patterns
        if theta_sym.shape[0] > 1:
            shuffled = torch.randperm(theta_sym.shape[0], device=theta_sym.device)
            balanced_ids = pattern_ids.clone()
            pattern_ids = torch.empty_like(balanced_ids)
            pattern_ids[shuffled] = balanced_ids

        theta_sym[:, dims] = theta_sym[:, dims] * sign_patterns[pattern_ids]
        return theta_sym

    def _build_sign_patterns(self, dims, dtype, device):
        dims = list(dims)
        n_patterns = 1 << len(dims)
        sign_patterns = torch.ones(
            n_patterns,
            len(dims),
            dtype=dtype,
            device=device,
        )
        for pattern_idx in range(n_patterns):
            for bit_idx in range(len(dims)):
                if (pattern_idx >> bit_idx) & 1:
                    sign_patterns[pattern_idx, bit_idx] = -1.0
        return sign_patterns

    def _apply_fixed_sign_pattern(self, theta_samples, dims, sign_pattern):
        if not dims:
            return theta_samples
        theta_fixed = theta_samples.clone()
        theta_fixed[:, dims] = theta_fixed[:, dims].abs() * sign_pattern.unsqueeze(0)
        return theta_fixed

    def _canonicalize_for_covariance(self, theta_samples):
        if not self.covariance_symmetry_dims:
            return theta_samples
        if theta_samples.ndim != 2 or theta_samples.shape[0] == 0:
            return theta_samples

        theta_canon = theta_samples.clone()
        dims = list(self.covariance_symmetry_dims)
        theta_canon[:, dims] = theta_canon[:, dims].abs()
        return theta_canon

    def compute_dsm_loss(self, theta_0, x):
        theta_0 = self.normalize_theta(theta_0)
        x = self.normalize_x(x)
        t = torch.rand(theta_0.shape[0], device=theta_0.device)
        eps = torch.randn_like(theta_0)
        theta_t = torch.sqrt(self.alpha(t)).unsqueeze(-1) * theta_0 + self.sigma(t).unsqueeze(-1) * eps
        pred_eps = self(theta_t, x, t)
        return F.mse_loss(pred_eps, eps)

    def _raw_log_prior(self, theta_raw):
        theta_raw = theta_raw.to(device=self.prior_lower.device, dtype=torch.float32)
        if theta_raw.ndim == 1:
            theta_raw = theta_raw.unsqueeze(0)

        lower = self.prior_lower.unsqueeze(0)
        upper = self.prior_upper.unsqueeze(0)
        in_bounds = ((theta_raw >= lower) & (theta_raw <= upper)).all(dim=1)
        neg_large = torch.full(
            (theta_raw.shape[0],),
            fill_value=-1e30,
            dtype=theta_raw.dtype,
            device=theta_raw.device,
        )
        if self.prior_type != "conditional":
            return torch.where(in_bounds, torch.zeros_like(neg_large), neg_large)

        sigma = torch.tensor(0.1, dtype=theta_raw.dtype, device=theta_raw.device)
        mu = theta_raw[:, 0].pow(2)
        theta_1 = theta_raw[:, 1]
        a = (theta_1 - mu) / sigma
        alpha = (self.prior_lower[1] - mu) / sigma
        beta = (self.prior_upper[1] - mu) / sigma
        sqrt_two = torch.tensor(math.sqrt(2.0), dtype=theta_raw.dtype, device=theta_raw.device)
        z = 0.5 * (1.0 + torch.erf(beta / sqrt_two)) - 0.5 * (1.0 + torch.erf(alpha / sqrt_two))
        z = torch.clamp(z, min=1e-30)
        log_full = (
            -0.5 * a.pow(2)
            - torch.log(sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - torch.log(z)
        )
        return torch.where(in_bounds, log_full, neg_large)

    def _uniform_prior_score(self, theta_t, t):
        t = self._normalize_time_input(t, batch_size=theta_t.shape[0], device=theta_t.device)
        sqrt_alpha = torch.sqrt(self.alpha(t)).unsqueeze(-1)
        sigma_t = self.sigma(t).unsqueeze(-1)

        upper_arg = (sqrt_alpha * self.prior_upper_norm.unsqueeze(0) - theta_t) / sigma_t
        lower_arg = (sqrt_alpha * self.prior_lower_norm.unsqueeze(0) - theta_t) / sigma_t

        cdf_upper = 0.5 * (1.0 + torch.erf(upper_arg / math.sqrt(2.0)))
        cdf_lower = 0.5 * (1.0 + torch.erf(lower_arg / math.sqrt(2.0)))
        cdf_diff = torch.clamp(cdf_upper - cdf_lower, min=1e-6)

        pdf_upper = torch.exp(-0.5 * upper_arg.pow(2)) / math.sqrt(2.0 * math.pi)
        pdf_lower = torch.exp(-0.5 * lower_arg.pow(2)) / math.sqrt(2.0 * math.pi)
        score = (-(pdf_upper - pdf_lower) / sigma_t) / cdf_diff
        return torch.clamp(score, -self.prior_score_clip, self.prior_score_clip)

    def _conditional_prior_score(self, theta_t, t):
        t = self._normalize_time_input(t, batch_size=theta_t.shape[0], device=theta_t.device)
        with torch.enable_grad():
            theta_req = theta_t if theta_t.requires_grad else theta_t.detach().clone().requires_grad_(True)
            sqrt_alpha = torch.sqrt(self.alpha(t)).unsqueeze(-1)
            z0_proxy = theta_req / torch.clamp(sqrt_alpha, min=1e-6)
            theta_raw_proxy = self.unnormalize_theta(z0_proxy)
            theta_raw_proxy = torch.clamp(theta_raw_proxy, self.prior_lower, self.prior_upper)
            log_prior = self._raw_log_prior(theta_raw_proxy)
            grad = torch.autograd.grad(
                log_prior.sum(),
                theta_req,
                create_graph=theta_req.requires_grad,
                retain_graph=theta_req.requires_grad,
            )[0]
        return grad if theta_t.requires_grad else grad.detach()

    def _prior_score_impl(self, theta_t, t):
        if self.prior_type == "conditional":
            return self._conditional_prior_score(theta_t, t)
        return self._uniform_prior_score(theta_t, t)

    @torch.no_grad()
    def prior_score(self, theta_t, t):
        return self._prior_score_impl(theta_t, t)

    def _pairwise_score(self, theta, x_obs, t):
        n_samples = theta.shape[0]
        n_obs = x_obs.shape[0]
        theta_rep = theta[:, None, :].expand(-1, n_obs, -1).reshape(n_samples * n_obs, self.theta_dim)
        x_rep = x_obs[None, :, :].expand(n_samples, -1, -1).reshape(n_samples * n_obs, self.x_dim)
        t_rep = self._normalize_time_input(t, batch_size=n_samples * n_obs, device=theta.device)
        scores = self.score(theta_rep, x_rep, t_rep)
        return scores.reshape(n_samples, n_obs, self.theta_dim)

    def _pairwise_score_jacobian(self, theta, x_obs, t):
        t = self._normalize_time_input(t, batch_size=theta.shape[0], device=theta.device)
        scores = []
        jacobians = []
        for sample_idx in range(theta.shape[0]):
            theta_i = theta[sample_idx].detach().clone().requires_grad_(True)
            score_rows = []
            jac_rows_all = []
            for obs_idx in range(x_obs.shape[0]):
                x_j = x_obs[obs_idx : obs_idx + 1]
                t_i = t[sample_idx : sample_idx + 1]
                score_ij = self.score(theta_i.unsqueeze(0), x_j, t_i).squeeze(0)
                score_rows.append(score_ij.detach())
                jac_rows = []
                for dim_idx in range(self.theta_dim):
                    grad_dim = torch.autograd.grad(
                        score_ij[dim_idx],
                        theta_i,
                        retain_graph=(dim_idx + 1) < self.theta_dim,
                        create_graph=False,
                    )[0]
                    jac_rows.append(grad_dim)
                jac_rows_all.append(torch.stack(jac_rows, dim=0))
            scores.append(torch.stack(score_rows, dim=0))
            jacobians.append(torch.stack(jac_rows_all, dim=0))
        return torch.stack(scores, dim=0), torch.stack(jacobians, dim=0)

    def tweedies_approximation(self, x_obs, theta, t, dist_cov_est=None, mode="GAUSS"):
        mode = str(mode).upper()
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        if mode == "JAC":
            scores, jac_score = self._pairwise_score_jacobian(theta, x_obs, t)
            eye = torch.eye(self.theta_dim, dtype=theta.dtype, device=theta.device)
            cov = (sigma_t.pow(2) / alpha_t) * (eye + sigma_t.pow(2) * jac_score)
            cov = 0.5 * (cov + cov.transpose(-1, -2)) + self.cov_reg * eye
            prec = torch.linalg.inv(cov)
        elif mode == "GAUSS":
            if dist_cov_est is None:
                raise ValueError("dist_cov_est is required when cov_mode='GAUSS'.")
            scores = self._pairwise_score(theta, x_obs, t)
            prec = prec_matrix_backward(t=t, dist_cov=dist_cov_est, model=self)
            prec = prec.unsqueeze(0).expand(theta.shape[0], -1, -1, -1)
        else:
            raise NotImplementedError("Gaussian benchmark currently supports cov_mode='GAUSS' or 'JAC'.")

        mean = (theta[:, None, :] + sigma_t.pow(2) * scores) / torch.sqrt(alpha_t)
        return prec, mean, scores

    def tweedies_approximation_prior(self, theta_t, t):
        with torch.enable_grad():
            theta_req = theta_t.detach().clone().requires_grad_(True)
            t = self._normalize_time_input(t, batch_size=theta_req.shape[0], device=theta_req.device)
            prior_score = self._prior_score_impl(theta_req, t)
            jac_rows = []
            for dim_idx in range(self.theta_dim):
                grad_dim = torch.autograd.grad(
                    prior_score[:, dim_idx].sum(),
                    theta_req,
                    retain_graph=(dim_idx + 1) < self.theta_dim,
                    create_graph=False,
                )[0]
                jac_rows.append(grad_dim)
            jac_score = torch.stack(jac_rows, dim=1)
            alpha_t = self.alpha(t).unsqueeze(-1).unsqueeze(-1)
            sigma_t = self.sigma(t).unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(self.theta_dim, dtype=theta_req.dtype, device=theta_req.device).unsqueeze(0)
            mean = (theta_req + self.sigma(t).unsqueeze(-1).pow(2) * prior_score) / torch.sqrt(
                self.alpha(t)
            ).unsqueeze(-1)
            cov = (sigma_t.pow(2) / alpha_t) * (eye + sigma_t.pow(2) * jac_score)
            cov = 0.5 * (cov + cov.transpose(-1, -2)) + self.cov_reg * eye
            return torch.linalg.inv(cov), mean.detach(), prior_score.detach()

    def prior_precision_tweedie(self, theta_t, t):
        prec, _, _ = self.tweedies_approximation_prior(theta_t, t)
        return prec

    def factorized_score(
        self,
        theta,
        x_obs,
        t,
        dist_cov_est,
        prior_cov_est=None,
        cov_mode=None,
    ):
        cov_mode = self.cov_mode if cov_mode is None else str(cov_mode).upper()
        n_obs = x_obs.shape[0]
        if self.prior_type == "conditional" and n_obs > 1 and not self._conditional_prior_warning_emitted:
            warnings.warn(
                "Using TD-NPSE factorized tall posterior composition with prior_type='conditional'. "
                "This is not an analytic diffused-prior setting, so the prior correction term is "
                "not theoretically valid here. For n_obs > 1, prefer uniform prior or retrain with "
                "CFG if you want a learned prior score surrogate; note that CFG baselines are usually weak.",
                stacklevel=2,
            )
            self._conditional_prior_warning_emitted = True
        prec_0_t, _, scores = self.tweedies_approximation(
            x_obs=x_obs,
            theta=theta,
            t=t,
            dist_cov_est=dist_cov_est,
            mode=cov_mode,
        )
        prior_score = self.prior_score(theta, t)
        total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)

        if self.tall_score_mode == "simple_sum":
            return torch.clamp(total_score, -self.tall_score_clip, self.tall_score_clip)
        if self.tall_score_mode not in {"gauss_refine", "gauss", "gauss_clip"}:
            raise ValueError(
                f"Unknown tall_score_mode='{self.tall_score_mode}'. "
                "Expected 'simple_sum', 'gauss_refine', 'gauss', or 'gauss_clip'."
            )

        if prior_cov_est is not None:
            prec_prior_0_t = prec_matrix_backward(t=t, dist_cov=prior_cov_est, model=self)
            if prec_prior_0_t.ndim == 2:
                prec_prior_0_t = prec_prior_0_t.unsqueeze(0).expand(theta.shape[0], -1, -1)
            prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
            prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
            lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
            weighted_scores = (1 - n_obs) * prec_score_prior + prec_score_post.sum(dim=1)
            eye = torch.eye(self.theta_dim, dtype=theta.dtype, device=theta.device).unsqueeze(0)
            lda = 0.5 * (lda + lda.transpose(-1, -2)) + self.cov_reg * eye
            total_score = torch.linalg.solve(lda, weighted_scores)
        elif n_obs > 1 and float(torch.sqrt(self.alpha(t)).detach().cpu().item()) > 0.5:
            prec_prior_0_t, _, _ = self.tweedies_approximation_prior(theta, t)
            prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
            prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
            lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
            weighted_scores = prec_score_prior + (prec_score_post - prec_score_prior[:, None]).sum(dim=1)
            eye = torch.eye(self.theta_dim, dtype=theta.dtype, device=theta.device).unsqueeze(0)
            lda = 0.5 * (lda + lda.transpose(-1, -2)) + self.cov_reg * eye
            total_score = torch.linalg.solve(lda, weighted_scores)
        return torch.clamp(total_score, -self.tall_score_clip, self.tall_score_clip)

    def diffused_tall_posterior_score(
        self,
        theta,
        x_obs,
        t,
        dist_cov_est=None,
        prior_cov_est=None,
        cov_mode=None,
    ):
        n_obs = x_obs.shape[0]
        if n_obs == 1:
            x_batch = x_obs.expand(theta.shape[0], -1)
            return self.score(theta, x_batch, t)
        return self.factorized_score(
            theta=theta,
            x_obs=x_obs,
            t=t,
            dist_cov_est=dist_cov_est,
            prior_cov_est=prior_cov_est,
            cov_mode=cov_mode,
        )

    @torch.no_grad()
    def ddim_step(self, theta_t, score_t, t_curr, t_prev, eta=None, clip_override=None):
        if eta is None:
            eta = self.ddim_eta
        alpha_t = self.alpha(t_curr)
        alpha_prev = self.alpha(t_prev)
        sigma_t = self.sigma(t_curr)
        sigma_prev = self.sigma(t_prev)

        x0_hat = (theta_t + sigma_t.pow(2) * score_t) / torch.sqrt(alpha_t)
        eps_hat = (theta_t - torch.sqrt(alpha_t) * x0_hat) / sigma_t
        bridge_std = eta * torch.sqrt(
            torch.clamp((sigma_prev.pow(2) / sigma_t.pow(2)) * (1.0 - alpha_t / alpha_prev), min=0.0)
        )
        coeff = torch.sqrt(torch.clamp(sigma_prev.pow(2) - bridge_std.pow(2), min=0.0))
        noise = (
            torch.randn_like(theta_t)
            if float(bridge_std.detach().cpu().item()) > 0.0
            else torch.zeros_like(theta_t)
        )
        theta_prev = torch.sqrt(alpha_prev) * x0_hat + coeff * eps_hat + bridge_std * noise
        clip_denoised = self.clip_denoised if clip_override is None else bool(clip_override)
        if clip_denoised:
            theta_prev = theta_prev.clamp(self.prior_lower_norm, self.prior_upper_norm)
        return theta_prev

    def _build_inference_schedule(self, num_steps, device):
        num_steps = max(1, int(num_steps))
        return torch.linspace(1.0, 0.0, steps=num_steps + 1, dtype=torch.float32, device=device)

    def _posterior_score(self, theta_t, x_obs_norm, t, covariances_0=None, prior_covariance_0=None):
        return self.diffused_tall_posterior_score(
            theta=theta_t,
            x_obs=x_obs_norm,
            t=t,
            dist_cov_est=covariances_0,
            prior_cov_est=prior_covariance_0,
            cov_mode=self.cov_mode,
        )

    @torch.no_grad()
    def euler_step(self, theta_t, score_t, t_curr, t_prev, clip_override=None):
        beta_t = self.beta(t_curr)
        dt = t_prev - t_curr
        drift = -0.5 * beta_t * theta_t - beta_t * score_t
        diffusion = torch.sqrt(torch.clamp(beta_t, min=0.0))
        theta_prev = theta_t + drift * dt + diffusion * torch.randn_like(theta_t) * torch.sqrt(dt.abs())
        clip_denoised = self.clip_denoised if clip_override is None else bool(clip_override)
        if clip_denoised:
            theta_prev = theta_prev.clamp(self.prior_lower_norm, self.prior_upper_norm)
        return theta_prev

    def _resolve_sample_steps(self, n_obs):
        del n_obs
        return max(1, int(self.sample_steps))

    @torch.no_grad()
    def _sample_posterior_normalized(
        self,
        x_obs_norm,
        n_samples,
        num_steps,
        covariances_0=None,
        prior_covariance_0=None,
        sign_pattern=None,
        clip_override=None,
    ):
        device = self.prior_lower.device
        theta_t = torch.randn(n_samples, self.theta_dim, device=device)
        if sign_pattern is not None and self.symmetry_mode_sampling_dims:
            theta_t = self._apply_fixed_sign_pattern(
                theta_t,
                list(self.symmetry_mode_sampling_dims),
                sign_pattern.to(device=device, dtype=theta_t.dtype),
            )

        step_indices = self._build_inference_schedule(num_steps, device)
        for t_curr, t_prev in zip(step_indices[:-1], step_indices[1:]):
            score_t = self._posterior_score(
                theta_t=theta_t,
                x_obs_norm=x_obs_norm,
                t=t_curr,
                covariances_0=covariances_0,
                prior_covariance_0=prior_covariance_0,
            )
            if self.sampler_type == "ddim":
                theta_t = self.ddim_step(theta_t, score_t, t_curr, t_prev, clip_override=clip_override)
            else:
                theta_t = self.euler_step(theta_t, score_t, t_curr, t_prev, clip_override=clip_override)
            if sign_pattern is not None and self.symmetry_mode_sampling_dims:
                theta_t = self._apply_fixed_sign_pattern(
                    theta_t,
                    list(self.symmetry_mode_sampling_dims),
                    sign_pattern.to(device=device, dtype=theta_t.dtype),
                )
        return theta_t

    @torch.no_grad()
    def _sample_single_posterior_normalized(
        self,
        x_single,
        n_samples,
        num_steps=None,
        eta=None,
        clip_override=None,
    ):
        self.eval()
        device = self.prior_lower.device
        if num_steps is None:
            num_steps = self.sample_steps
        step_indices = self._build_inference_schedule(num_steps, device)

        x_single = torch.as_tensor(x_single, dtype=torch.float32, device=device)
        if x_single.ndim == 1:
            x_single = x_single.unsqueeze(0)
        x_batch = self.normalize_x(x_single).expand(n_samples, -1)
        theta_t = torch.randn(n_samples, self.theta_dim, device=device)

        for t_curr, t_prev in zip(step_indices[:-1], step_indices[1:]):
            score_t = self.score(theta_t, x_batch, t_curr)
            theta_t = self.ddim_step(
                theta_t,
                score_t,
                t_curr,
                t_prev,
                eta=eta,
                clip_override=clip_override,
            )
        return theta_t

    @torch.no_grad()
    def sample_single_posterior(self, x_single, n_samples, num_steps=None, eta=None):
        samples_norm = self._sample_single_posterior_normalized(
            x_single=x_single,
            n_samples=n_samples,
            num_steps=num_steps,
            eta=eta,
        )
        samples_norm = self._apply_sign_flip_symmetry(samples_norm)
        return self.unnormalize_theta(samples_norm).cpu().numpy()

    @torch.no_grad()
    def _sample_prior_normalized(self, n_samples, num_steps=None, eta=0.5, clip_override=False):
        self.eval()
        device = self.prior_lower.device
        if num_steps is None:
            num_steps = self.cov_steps
        step_indices = self._build_inference_schedule(num_steps, device)
        theta_t = torch.randn(n_samples, self.theta_dim, device=device)
        for t_curr, t_prev in zip(step_indices[:-1], step_indices[1:]):
            prior_score_t = self.prior_score(theta_t, t_curr)
            theta_t = self.ddim_step(
                theta_t,
                prior_score_t,
                t_curr,
                t_prev,
                eta=eta,
                clip_override=clip_override,
            )
        return theta_t

    @torch.no_grad()
    def estimate_single_covariances(self, x_obs, n_samples=None, num_steps=None):
        self.eval()
        x_obs = torch.as_tensor(x_obs, dtype=torch.float32, device=self.prior_lower.device)
        if x_obs.ndim == 3:
            x_obs = x_obs.squeeze(0)
        if n_samples is None:
            n_samples = self.cov_samples
        if num_steps is None:
            num_steps = self.cov_steps

        covariances = []
        eye = torch.eye(self.theta_dim, dtype=torch.float32, device=self.prior_lower.device)
        for j in range(x_obs.shape[0]):
            samples_norm = self._sample_single_posterior_normalized(
                x_obs[j],
                n_samples=n_samples,
                num_steps=num_steps,
                eta=0.5,
                clip_override=False,
            )
            samples_norm = self._canonicalize_for_covariance(samples_norm)
            cov = torch.cov(samples_norm.T) * self.covariance_scale
            covariances.append(cov + self.cov_reg * eye)
        return torch.stack(covariances, dim=0)

    @torch.no_grad()
    def estimate_prior_covariance(self, n_samples=None, num_steps=None):
        self.eval()
        if n_samples is None:
            n_samples = self.cov_samples
        if num_steps is None:
            num_steps = self.cov_steps
        eye = torch.eye(self.theta_dim, dtype=torch.float32, device=self.prior_lower.device)
        samples_norm = self._sample_prior_normalized(
            n_samples=n_samples,
            num_steps=num_steps,
            eta=0.5,
            clip_override=False,
        )
        cov = torch.cov(samples_norm.T) * self.covariance_scale
        return cov + self.cov_reg * eye

    @torch.no_grad()
    def sample_posterior(
        self,
        x_obs,
        n_samples,
        num_steps=None,
        covariances_0=None,
        prior_covariance_0=None,
    ):
        self.eval()
        device = self.prior_lower.device

        x_obs = torch.as_tensor(x_obs, dtype=torch.float32, device=device)
        if x_obs.ndim == 3:
            x_obs = x_obs.squeeze(0)
        n_obs = x_obs.shape[0]
        if num_steps is None:
            num_steps = self._resolve_sample_steps(n_obs)

        x_obs_norm = self.normalize_x(x_obs)
        if self.cov_mode == "GAUSS":
            if covariances_0 is None:
                covariances_0 = self.estimate_single_covariances(x_obs)
            covariances_0 = covariances_0.to(device=device, dtype=torch.float32)
            if prior_covariance_0 is None and self.prior_type == "uniform":
                prior_covariance_0 = self.estimate_prior_covariance(
                    n_samples=self.cov_samples,
                    num_steps=self.cov_steps,
                )
            if prior_covariance_0 is not None:
                prior_covariance_0 = prior_covariance_0.to(device=device, dtype=torch.float32)

        if self.symmetry_mode_sampling_dims:
            dims = list(self.symmetry_mode_sampling_dims)
            sign_patterns = self._build_sign_patterns(dims, dtype=torch.float32, device=device)
            n_patterns = int(sign_patterns.shape[0])
            base = n_samples // n_patterns
            remainder = n_samples % n_patterns
            chunks = []
            for pattern_idx in range(n_patterns):
                chunk_size = base + (1 if pattern_idx < remainder else 0)
                if chunk_size <= 0:
                    continue
                chunk = self._sample_posterior_normalized(
                    x_obs_norm=x_obs_norm,
                    n_samples=chunk_size,
                    num_steps=num_steps,
                    covariances_0=covariances_0,
                    prior_covariance_0=prior_covariance_0,
                    sign_pattern=sign_patterns[pattern_idx],
                )
                chunks.append(chunk)
            if not chunks:
                theta_t = torch.empty((0, self.theta_dim), dtype=torch.float32, device=device)
            else:
                theta_t = torch.cat(chunks, dim=0)
                if theta_t.shape[0] > 1:
                    theta_t = theta_t[torch.randperm(theta_t.shape[0], device=device)]
        else:
            theta_t = self._sample_posterior_normalized(
                x_obs_norm=x_obs_norm,
                n_samples=n_samples,
                num_steps=num_steps,
                covariances_0=covariances_0,
                prior_covariance_0=prior_covariance_0,
                sign_pattern=None,
            )
        theta_t = self._apply_sign_flip_symmetry(theta_t)
        return self.unnormalize_theta(theta_t).cpu().numpy()


class RandomSingleObservationDataset(Dataset):
    def __init__(self, theta_train, x_train, pairs_per_theta=1):
        self.theta_train = np.asarray(theta_train, dtype=np.float32)
        self.x_train = np.asarray(x_train, dtype=np.float32)
        self.pairs_per_theta = int(max(1, pairs_per_theta))
        if self.x_train.ndim != 3:
            raise ValueError(f"Expected x_train with shape (N, n, d_x), got {self.x_train.shape}")
        if self.theta_train.ndim != 2:
            raise ValueError(f"Expected theta_train with shape (N, d), got {self.theta_train.shape}")
        if self.theta_train.shape[0] != self.x_train.shape[0]:
            raise ValueError("theta_train and x_train must share the same first dimension.")

    def __len__(self):
        return self.theta_train.shape[0] * self.pairs_per_theta

    def __getitem__(self, idx):
        idx = idx % self.theta_train.shape[0]
        obs_idx = np.random.randint(self.x_train.shape[1])
        return (
            torch.from_numpy(self.x_train[idx, obs_idx]),
            torch.from_numpy(self.theta_train[idx]),
        )


class FullSingleObservationDataset(Dataset):
    def __init__(self, theta_train, x_train):
        self.theta_train = np.asarray(theta_train, dtype=np.float32)
        self.x_train = np.asarray(x_train, dtype=np.float32)
        if self.x_train.ndim != 3:
            raise ValueError(f"Expected x_train with shape (N, n, d_x), got {self.x_train.shape}")
        if self.theta_train.ndim != 2:
            raise ValueError(f"Expected theta_train with shape (N, d), got {self.theta_train.shape}")
        if self.theta_train.shape[0] != self.x_train.shape[0]:
            raise ValueError("theta_train and x_train must share the same first dimension.")
        self.x_single, self.theta_single = build_single_observation_dataset(self.theta_train, self.x_train)

    def __len__(self):
        return self.theta_single.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x_single[idx]), torch.from_numpy(self.theta_single[idx])


def build_single_observation_dataset(theta_train, x_train):
    theta_train = np.asarray(theta_train, dtype=np.float32)
    x_train = np.asarray(x_train, dtype=np.float32)
    n_sims, n_obs, x_dim = x_train.shape
    theta_single = np.repeat(theta_train, repeats=n_obs, axis=0)
    x_single = x_train.reshape(n_sims * n_obs, x_dim)
    return x_single, theta_single


def train_td_npse(
    model,
    train_loader,
    epochs,
    device,
    learning_rate=1e-4,
    lr_schedule="none",
    eta_min=0.0,
    validation_split=0.2,
    early_stopping=True,
    patience=200,
    min_nb_epochs=100,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule_name = str(lr_schedule).lower()
    scheduler = None
    if schedule_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(eta_min),
        )

    model.to(device)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
    )
    ema_model.to(device)

    dataset = train_loader.dataset
    batch_size = getattr(train_loader, "batch_size", None) or 256
    if validation_split > 0.0:
        n_total = len(dataset)
        n_val = max(1, int(round(n_total * validation_split)))
        n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        train_subset, val_subset = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_iter = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_iter = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        train_iter = train_loader
        val_iter = None
        n_train = len(dataset)
        n_val = 0

    loss_history = []
    val_history = []
    best_loss = None
    best_state = None
    best_epoch = None

    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])
        for x_batch, theta_batch in train_iter:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = model.compute_dsm_loss(theta_batch, x_batch)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "TD-NPSE training loss does not require grad. "
                        "This usually means training is running under a global no_grad/inference_mode context "
                        "or model parameters have requires_grad=False."
                    )
                loss.backward()
                optimizer.step()
                ema_model.update_parameters(model)
            epoch_loss += loss.item() * theta_batch.shape[0]

        avg_loss = epoch_loss / max(n_train, 1)
        loss_history.append(avg_loss)
        val_loss = None

        if val_iter is not None:
            ema_model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for x_batch, theta_batch in val_iter:
                    x_batch = x_batch.to(device)
                    theta_batch = theta_batch.to(device)
                    loss = ema_model.module.compute_dsm_loss(theta_batch, x_batch)
                    total_val_loss += loss.item() * theta_batch.shape[0]
            val_loss = total_val_loss / max(n_val, 1)
            val_history.append(val_loss)

            if early_stopping and epoch >= min_nb_epochs:
                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(ema_model.module.state_dict())
                elif best_epoch is not None and (epoch - best_epoch) > patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"best val loss {best_loss:.4f} at epoch {best_epoch + 1}"
                    )
                    break

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loss is None:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {current_lr:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Val: {val_loss:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(ema_model.module.state_dict())
    model.val_history = val_history
    model.best_epoch = best_epoch
    return loss_history
