"""Smoke tests for the shared SMMD components and per-experiment models."""

import numpy as np
import torch

from sbi_common.nn import Generator, RMSNorm, SMMD_Model
from sbi_common.losses import mmd_loss as shared_mmd_loss
from sbi_common.losses import sliced_mmd_loss as shared_sliced_mmd_loss


def test_rmsnorm_shape_and_stability():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16)
    out = RMSNorm(16)(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_generator_layer_counts():
    torch.manual_seed(0)
    gen3 = Generator(z_dim=5, stats_dim=10, out_dim=5, hidden_layers=3)
    gen4 = Generator(z_dim=5, stats_dim=10, out_dim=5, hidden_layers=4)
    # Each hidden block contributes one Linear layer; plus the output layer.
    assert sum(isinstance(m, torch.nn.Linear) for m in gen3.modules()) == 4
    assert sum(isinstance(m, torch.nn.Linear) for m in gen4.modules()) == 5
    z = torch.randn(2, 7, 5)
    stats = torch.randn(2, 10)
    assert gen3(z, stats).shape == (2, 7, 5)
    assert gen4(z, stats).shape == (2, 7, 5)


def _import_experiment_models(exp_dir):
    import importlib

    smmd = importlib.import_module(f"{exp_dir}.models.smmd")
    mmd = importlib.import_module(f"{exp_dir}.models.mmd")
    dnnabc = importlib.import_module(f"{exp_dir}.models.dnnabc")
    return smmd, mmd, dnnabc


def test_gaussian_models_smoke():
    torch.manual_seed(0)
    smmd, mmd, _ = _import_experiment_models("Gaussian")
    model = smmd.SMMD_Model(summary_dim=6, d=3, d_x=2, n=5)
    x = torch.randn(2, 5, 2)
    z = torch.randn(2, 4, 3)
    out = model(x, z)
    assert out.shape == (2, 4, 3)
    samples = model.sample_posterior(x[0].numpy(), n_samples=11)
    assert samples.shape == (11, 3)
    mmd_model = mmd.MMD_Model(summary_dim=6, d=3, d_x=2, n=5)
    assert isinstance(mmd_model, smmd.SMMD_Model)


def test_gaussian_mmd_loss_is_normalized():
    """The Gaussian task uses normalized kernels, like the other tasks."""
    torch.manual_seed(0)
    smmd, mmd, _ = _import_experiment_models("Gaussian")
    theta_true = torch.randn(2, 3)
    theta_fake = torch.randn(2, 6, 3)
    got = mmd.mmd_loss(theta_true, theta_fake, n_obs=50)
    expected = shared_mmd_loss(theta_true, theta_fake, scale=50, normalize=True)
    assert torch.allclose(got, expected)
    # Sanity check: the value differs from the unnormalized variant.
    unnormalized = shared_mmd_loss(theta_true, theta_fake, scale=50, normalize=False)
    assert not torch.allclose(got, unnormalized)


def test_gaussian_loss_bandwidth_tracks_observation_count():
    """Gaussian MMD/SMMD bandwidths scale as k^2 / n_obs, not with M=50."""
    torch.manual_seed(0)
    smmd, mmd, _ = _import_experiment_models("Gaussian")
    theta_true = torch.randn(2, 3)
    theta_fake = torch.randn(2, 6, 3)
    for n_obs in (25, 50, 100):
        got_mmd = mmd.mmd_loss(theta_true, theta_fake, n_obs=n_obs)
        expected_mmd = shared_mmd_loss(theta_true, theta_fake, scale=n_obs, normalize=True)
        assert torch.allclose(got_mmd, expected_mmd)
        # Both variants draw random projections internally; reseed so the
        # wrapper and the shared implementation use the same projections.
        torch.manual_seed(7)
        got_smmd = smmd.sliced_mmd_loss(theta_true, theta_fake, num_slices=8, n_obs=n_obs)
        torch.manual_seed(7)
        expected_smmd = shared_sliced_mmd_loss(theta_true, theta_fake, num_slices=8, scale=n_obs)
        assert torch.allclose(got_smmd, expected_smmd)
    # The bandwidth is not frozen at the M=50 value: n=100 must give a
    # different (narrower-kernel) loss than n=50.
    torch.manual_seed(1)
    theta_true2 = torch.randn(2, 3)
    theta_fake2 = torch.randn(2, 6, 3)
    assert not torch.allclose(
        mmd.mmd_loss(theta_true2, theta_fake2, n_obs=50),
        mmd.mmd_loss(theta_true2, theta_fake2, n_obs=100),
    )


def test_lotka_models_scaling():
    torch.manual_seed(0)
    smmd, _, _ = _import_experiment_models("LotkaVoletrra")
    lower = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    model = smmd.SMMD_Model(summary_dim=4, d=5, d_x=2, n=151, theta_lower=lower, theta_upper=upper)
    x = torch.randn(2, 151, 2)
    z = torch.randn(2, 3, 5)
    scaled = model(x, z)
    assert scaled.abs().max().item() <= 1.0 + 1e-6  # tanh output in [-1, 1]
    samples = model.sample_posterior(x[0].numpy(), n_samples=10)
    assert samples.shape == (10, 5)
    assert samples.min() >= lower.min() - 1e-5 and samples.max() <= upper.max() + 1e-5


def test_radio_models_standardizer():
    torch.manual_seed(0)
    smmd, _, _ = _import_experiment_models("Radio")
    ns, n_real = 801, 4
    model = smmd.SMMD_Model(summary_dim=4, d=4, d_x=1, n=n_real)
    mean = torch.tensor([10.0, 20.0, 30.0, 40.0])
    std = torch.tensor([1.0, 2.0, 3.0, 4.0])
    model.set_theta_standardizer(mean, std)
    x = torch.randn(2, n_real, ns)
    samples = model.sample_posterior(x[0].numpy(), n_samples=9)
    assert samples.shape == (9, 4)
    # Without the standardizer the output stays on the raw generator scale.
    model2 = smmd.SMMD_Model(summary_dim=4, d=4, d_x=1, n=n_real)
    samples2 = model2.sample_posterior(x[0].numpy(), n_samples=9)
    assert not np.allclose(samples, samples2)


def test_sir_models_smoke():
    torch.manual_seed(0)
    smmd, mmd, dnnabc = _import_experiment_models("SIR")
    model = smmd.SMMD_Model(summary_dim=4, d=2, d_x=1, n=20)
    x = torch.randn(2, 20, 1)
    z = torch.randn(2, 4, 2)
    assert model(x, z).shape == (2, 4, 2)
    assert model.sample_posterior(x[0].numpy(), n_samples=5).shape == (5, 2)
    assert isinstance(mmd.MMD_Model(summary_dim=4, d=2, d_x=1, n=20), smmd.SMMD_Model)
    dn = dnnabc.DNNABC_Model(d=2, d_x=1)
    assert dn(x).shape == (2, 2)
