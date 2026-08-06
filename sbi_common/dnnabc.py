"""DNNABC: ABC rejection sampling with a learned (regression) summary network.

The training loop and rejection sampler are shared by all experiments; the
per-experiment ``models/dnnabc.py`` modules supply the task-specific summary
network and bind the experiment's hyperparameters (optimizer, scheduler,
target transform, forward chunking).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DNNABC_Model(nn.Module):
    """Regressor that maps observations to theta (learned summary statistics)."""

    def __init__(self, summary_net, d):
        super().__init__()
        self.net = summary_net
        self.d = d

    def forward(self, x):
        # x: (batch, n_points, d_x) -> (batch, d) predicted theta.
        return self.net(x)


def train_dnnabc(
    model,
    train_loader,
    epochs,
    device,
    *,
    lr=3e-4,
    optimizer="adamw",
    use_cosine_scheduler=True,
    theta_target_transform=None,
    log_every=10,
):
    """Train a DNNABC regression model with MSE loss.

    Args:
        model: :class:`DNNABC_Model` (or compatible nn.Module).
        train_loader: DataLoader yielding ``(x_batch, theta_batch)``.
        epochs: Number of training epochs.
        device: Torch device string.
        lr: Learning rate.
        optimizer: ``"adam"`` or ``"adamw"``.
        use_cosine_scheduler: Apply a cosine annealing LR schedule.
        theta_target_transform: Optional callable applied to ``theta_batch``
            before computing the loss (e.g. log-transform for positive theta).
        log_every: Print training loss every ``log_every`` epochs
            (``None`` disables printing).

    Returns:
        List of per-epoch mean losses.
    """
    optimizer_cls = optim.Adam if optimizer == "adam" else optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    scheduler = None
    if use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    loss_history = []
    print("Starting training (DNNABC)...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            if theta_target_transform is not None:
                theta_batch = theta_target_transform(theta_batch)

            optimizer.zero_grad()
            with torch.enable_grad():
                theta_pred = model(x_batch)
                loss = criterion(theta_pred, theta_batch)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if log_every is not None:
            is_log_epoch = (epoch + 1) % log_every == 0
            if epoch == 0 or epoch == epochs - 1 or is_log_epoch:
                lr_now = scheduler.get_last_lr()[0] if scheduler is not None else lr
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {lr_now:.6f}")

    return loss_history


def abc_rejection_sampling(
    model,
    x_obs,
    task,
    n_samples=1000,
    n_pool=100000,
    batch_size=10000,
    device="cpu",
    *,
    preprocess_obs_fn=None,
    forward_chunk=None,
):
    """ABC rejection sampling using a trained DNNABC model as summary statistics.

    Computes ``S_obs = model(x_obs)``, simulates ``n_pool`` prior/simulator
    pairs in batches, scores them by the Euclidean distance between summaries,
    and keeps the ``n_samples`` closest draws.

    Args:
        model: Trained DNNABC model.
        x_obs: Observation ``(n, d_x)`` or ``(1, n, d_x)``.
        task: Object exposing ``sample_prior(batch_size)`` and
            ``simulator(theta)``.
        n_samples: Number of accepted posterior samples.
        n_pool: Total number of prior draws to simulate.
        batch_size: Simulation/score batch size.
        device: Torch device string.
        preprocess_obs_fn: Optional callable applied to simulated x batches
            before scoring (e.g. casting to a different representation).
        forward_chunk: If set, run the model forward pass in chunks of this
            size to bound peak memory.

    Returns:
        ``(n_samples, d)`` numpy array of accepted theta draws.
    """
    model.eval()

    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.from_numpy(x_obs).float()
    if x_obs.ndim == 2:
        x_obs = x_obs.unsqueeze(0)
    x_obs = x_obs.to(device)

    with torch.no_grad():
        s_obs = model(x_obs)  # (1, d)

    all_thetas = []
    all_distances = []

    print(f"DNNABC: Simulating {n_pool} samples for rejection...")

    for i in range(0, n_pool, batch_size):
        current_batch_size = min(batch_size, n_pool - i)

        theta_batch_np = task.sample_prior(current_batch_size)
        x_batch_np = task.simulator(theta_batch_np)
        if preprocess_obs_fn is not None:
            x_batch_np = preprocess_obs_fn(x_batch_np)

        if forward_chunk is not None:
            dists = np.empty((current_batch_size,), dtype=np.float32)
            for j in range(0, current_batch_size, forward_chunk):
                j1 = min(current_batch_size, j + forward_chunk)
                x_sub = torch.from_numpy(x_batch_np[j:j1]).float().to(device)
                with torch.no_grad():
                    s_sim = model(x_sub)  # (chunk, d)
                dists[j:j1] = torch.norm(s_sim - s_obs, dim=1).detach().cpu().numpy().astype(np.float32)
        else:
            x_batch = torch.from_numpy(x_batch_np).float().to(device)
            with torch.no_grad():
                s_sim = model(x_batch)  # (batch, d)
            dists = torch.norm(s_sim - s_obs, dim=1).cpu().numpy()

        all_thetas.append(theta_batch_np)
        all_distances.append(dists)

    all_thetas = np.concatenate(all_thetas, axis=0)
    all_distances = np.concatenate(all_distances, axis=0)

    # Keep the n_samples closest draws.
    idx_sorted = np.argsort(all_distances)
    accepted_idx = idx_sorted[:n_samples]
    accepted_samples = all_thetas[accepted_idx]
    eps = all_distances[accepted_idx[-1]]

    print(f"DNNABC: Accepted {n_samples} samples. Max Epsilon: {eps:.4f}")
    return accepted_samples
