import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import time

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class TurinSummarySBI(nn.Module):
    def __init__(self, ns: int = 801, input_dim: int = 1, out_dim: int = 8, hidden_dim: int = 4):
        super().__init__()
        self.ns = int(ns)
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(64, self.out_dim, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim == 2:
            x = x[:, None, :]
        b, n, ns = x.shape
        x = x.reshape(b * n, 1, ns)
        h = self.conv(x).squeeze(-1)
        h = h.reshape(b, n, -1).mean(dim=1)
        return self.proj(h)


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (n, d), y: (m, d)
    x = x.float()
    y = y.float()
    with torch.no_grad():
        dxx = torch.cdist(x, x).pow(2)
        med = torch.median(dxx[dxx > 0]) if torch.any(dxx > 0) else torch.tensor(1.0, device=x.device)
        sigma2 = torch.clamp(med, min=eps)

    kxx = torch.exp(-0.5 * dxx / sigma2)
    dyy = torch.cdist(y, y).pow(2)
    kyy = torch.exp(-0.5 * dyy / sigma2)
    dxy = torch.cdist(x, y).pow(2)
    kxy = torch.exp(-0.5 * dxy / sigma2)

    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def _make_loader(theta: torch.Tensor, x: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(theta, x)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_npe_rs(
    *,
    task,
    x_obs: np.ndarray,
    cfg: dict,
    theta_train_np: Optional[np.ndarray] = None,
    x_train_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    mc = dict(cfg.get("models_config", {}).get("npe_rs", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    low = torch.as_tensor(task.lower, dtype=torch.float32, device=device)
    high = torch.as_tensor(task.upper, dtype=torch.float32, device=device)
    prior = BoxUniform(low=low, high=high, device=str(device))

    use_prior_new = bool(mc.get("use_prior_new", False))
    if use_prior_new:
        low_new = torch.as_tensor(mc["prior_new_low"], dtype=torch.float32, device=device)
        high_new = torch.as_tensor(mc["prior_new_high"], dtype=torch.float32, device=device)
        prior_for_sampling = BoxUniform(low=low_new, high=high_new, device=str(device))
    else:
        prior_for_sampling = prior

    num_simulations = int(mc.get("num_simulations", 2000))
    batch_size = int(mc.get("training_batch_size", 50))
    lr = float(mc.get("learning_rate", 5e-4))
    max_epochs = int(mc.get("max_num_epochs", 200))
    val_frac = float(mc.get("validation_fraction", 0.1))
    stop_after = int(mc.get("stop_after_epochs", 20))
    beta = float(mc.get("beta", 1.0))
    distance = str(mc.get("distance", "mmd")).lower()

    ns = int(cfg.get("Ns", getattr(task, "Ns", 801)))
    embed_hidden_dim = int(mc.get("summary_hidden_dim", 4))
    embed_out_dim = int(mc.get("summary_out_dim", 8))
    hidden_features = int(mc.get("hidden_features", 100))
    num_transforms = int(mc.get("num_transforms", 5))

    embedding_net = TurinSummarySBI(ns=ns, input_dim=1, out_dim=embed_out_dim, hidden_dim=embed_hidden_dim).to("cpu")
    build_nn = posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_theta="independent",
        z_score_x="independent",
    )

    if theta_train_np is not None and x_train_np is not None:
        theta_np = np.asarray(theta_train_np, dtype=np.float32)[:num_simulations]
        x_np = np.asarray(x_train_np, dtype=np.float32)[:num_simulations]
    else:
        theta = prior.sample((num_simulations,))
        theta_np = theta.detach().cpu().numpy().astype(np.float32)
        x_np = task.simulator(theta_np).astype(np.float32)

    theta = torch.from_numpy(theta_np).to(device)
    x = torch.from_numpy(x_np).to(device)
    if x.ndim == 2:
        x = x.unsqueeze(-1)

    idx = torch.randperm(theta.shape[0], device=device)
    n_val = int(round(val_frac * theta.shape[0]))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    theta_train = theta[train_idx]
    x_train = x[train_idx]
    theta_val = theta[val_idx]
    x_val = x[val_idx]

    net = build_nn(theta_train[:batch_size].detach().cpu(), x_train[:batch_size].detach().cpu()).to(device)
    for p in net.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    x_obs_t = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), device=device)
    if x_obs_t.ndim == 1:
        x_obs_t = x_obs_t[None, None, :]
    if x_obs_t.ndim == 2:
        x_obs_t = x_obs_t[None, :, :]

    best_val = float("inf")
    bad_epochs = 0

    train_loader = _make_loader(theta_train, x_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(theta_val, x_val, batch_size=batch_size, shuffle=False)

    _log(
        f"NPE-RS (SBI) | epochs={max_epochs}, lr={lr}, batch_size={batch_size}, "
        f"beta={beta}, distance={distance}, device={device}"
    )
    t0 = time.perf_counter()
    for ep in range(max_epochs):
        net.train()
        ep_loss_sum = 0.0
        ep_base_sum = 0.0
        ep_reg_sum = 0.0
        n_train_batches = 0
        for th_b, x_b in train_loader:
            opt.zero_grad()
            with torch.enable_grad():
                loss_base = net.loss(th_b, x_b).mean()
                reg = None

                if beta > 0.0:
                    emb_x = net.embedding_net(x_b)
                    emb_obs = net.embedding_net(x_obs_t.expand(x_b.shape[0], -1, -1))
                    if distance == "mmd":
                        reg = _mmd_rbf(emb_x, emb_obs)
                    else:
                        reg = torch.mean((emb_x.mean(dim=0) - emb_obs.mean(dim=0)) ** 2)
                    loss = loss_base + float(beta) * reg
                else:
                    loss = loss_base
                if not loss.requires_grad:
                    raise RuntimeError("NPE-RS loss has no grad_fn; check that torch grad is enabled and net is on the correct device.")
                loss.backward()
                opt.step()
            ep_loss_sum += float(loss.detach().cpu().item())
            ep_base_sum += float(loss_base.detach().cpu().item())
            if reg is not None:
                ep_reg_sum += float(reg.detach().cpu().item())
            n_train_batches += 1

        net.eval()
        with torch.no_grad():
            losses = []
            for th_b, x_b in val_loader:
                losses.append(net.loss(th_b, x_b).mean().item())
            val_loss = float(np.mean(losses)) if losses else float("inf")

        if (ep + 1) % 10 == 0 or ep == 0 or ep == max_epochs - 1:
            elapsed = time.perf_counter() - t0
            denom = float(max(n_train_batches, 1))
            _log(
                f"NPE-RS epoch {ep + 1}/{max_epochs} | "
                f"train_loss={ep_loss_sum / denom:.6f} "
                f"(base={ep_base_sum / denom:.6f}, reg={ep_reg_sum / denom:.6f}) | "
                f"val_loss={val_loss:.6f} | best_val={best_val:.6f} | "
                f"bad_epochs={bad_epochs}/{stop_after} | elapsed={elapsed:.1f}s"
            )

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= stop_after:
                _log(f"NPE-RS early stop at epoch {ep + 1}/{max_epochs} | best_val={best_val:.6f}")
                break

    posterior = DirectPosterior(posterior_estimator=net, prior=prior_for_sampling, device=device, x_shape=x_obs_t.shape[1:])
    n_post = int(cfg.get("n_samples_posterior", 1000))
    with torch.no_grad():
        theta_post = posterior.sample((n_post,), x=x_obs_t, show_progress_bars=False)
    return theta_post.detach().cpu().numpy().astype(np.float32)
