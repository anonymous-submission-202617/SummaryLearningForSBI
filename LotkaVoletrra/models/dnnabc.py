"""DNNABC for the Lotka-Volterra task (CNN summary + ABC rejection)."""

from sbi_common.dnnabc import (
    DNNABC_Model as _DNNABC_Model,
    abc_rejection_sampling as _abc_rejection_sampling,
    train_dnnabc as _train_dnnabc,
)

from .smmd import SummaryNet


class DNNABC_Model(_DNNABC_Model):
    """Regressor mapping observations to theta via the CNN summary network."""

    def __init__(self, d=4, d_x=2):
        super().__init__(
            summary_net=SummaryNet(input_dim=d_x, output_dim=d),
            d=d,
        )


def train_dnnabc(model, train_loader, epochs, device):
    """Train with AdamW (lr=1e-3) and cosine LR scheduling (LV defaults)."""
    return _train_dnnabc(
        model,
        train_loader,
        epochs,
        device,
        lr=1e-3,
        optimizer="adamw",
        use_cosine_scheduler=True,
    )


def abc_rejection_sampling(
    model,
    x_obs,
    task,
    n_samples=1000,
    n_pool=100000,
    batch_size=10000,
    device="cpu",
    preprocess_obs_fn=None,
):
    """ABC rejection sampling with optional observation preprocessing."""
    return _abc_rejection_sampling(
        model,
        x_obs,
        task,
        n_samples=n_samples,
        n_pool=n_pool,
        batch_size=batch_size,
        device=device,
        preprocess_obs_fn=preprocess_obs_fn,
    )
