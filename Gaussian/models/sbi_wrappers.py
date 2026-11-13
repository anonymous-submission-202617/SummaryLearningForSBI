"""Thin wrapper around the ``sbi`` package for training and sampling the
SNPE-C / NPE estimators, together with the hand-crafted summary statistics
they consume."""

import torch
import numpy as np
try:
    # sbi >= 0.22 (incl. the pinned 0.26.1): SNPE* naming.
    from sbi.inference import SNPE_A, SNPE_B, SNPE as _SNPE_CLASS

    NPE, NPE_A, NPE_B = _SNPE_CLASS, SNPE_A, SNPE_B
except ImportError:
    # sbi <= 0.19: legacy NPE* naming.
    from sbi.inference import NPE, NPE_A, NPE_B
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

from sbi_common.checkpoints import build_sbi_estimator, restore_torch


def calculate_summary_statistics(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


def run_sbi_model(
    model_type,
    train_loader,
    x_obs,
    theta_true,
    task,
    device="cpu",
    num_rounds=1,
    sims_per_round=1000,
    max_epochs=1000,
    n_samples=1000,
    **kwargs,
):
    print(
        f"=== Running {model_type.upper()} (Rounds={num_rounds}, Sims/Round={sims_per_round}, MaxEpochs={max_epochs}) ==="
    )

    initial_training_data = kwargs.pop("initial_training_data", None)
    train_validation_fraction = float(kwargs.pop("validation_fraction", 0.1))
    train_stop_after_epochs = int(kwargs.pop("stop_after_epochs", 20))
    train_learning_rate = float(kwargs.pop("learning_rate", 5e-4))
    train_batch_size = int(kwargs.pop("training_batch_size", 200))
    show_train_summary = bool(kwargs.pop("show_train_summary", False))

    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()

    if x_obs.ndim == 2:
        x_obs_input = x_obs.unsqueeze(0)
    elif x_obs.ndim == 3:
        x_obs_input = x_obs
    else:
        raise ValueError(f"Expected x_obs shape (n, d) or (1, n, d), got {x_obs.shape}")

    x_obs_stats = calculate_summary_statistics(x_obs_input)
    print(f"Observation Summary Stats Shape: {x_obs_stats.shape}")

    # MPS has autograd gaps (linalg_qr etc.), force CPU. CUDA is fine.
    if device and "cuda" in str(device).lower():
        device_sbi = device
    elif device and "mps" in str(device).lower():
        print("  MPS detected → forcing CPU for SBI (MPS autograd not fully supported)")
        device_sbi = "cpu"
    elif torch.cuda.is_available():
        device_sbi = "cuda"
        print("  CUDA available → using GPU for SBI")
    else:
        device_sbi = "cpu"

    low = torch.tensor(task.lower, dtype=torch.float32, device=device_sbi)
    high = torch.tensor(task.upper, dtype=torch.float32, device=device_sbi)
    prior = BoxUniform(low=low, high=high, device=str(device_sbi))

    model_type_lower = model_type.lower()
    if model_type_lower == "snpe_a":
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type_lower == "snpe_b":
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type_lower in ["snpe_c", "npe", "snpe"]:
        inference = NPE(prior=prior, device=device_sbi)
    else:
        print(
            f"Warning: Unknown model_type '{model_type}'. Defaulting to NPE (SNPE-C)."
        )
        inference = NPE(prior=prior, device=device_sbi)

    checkpoint_path = kwargs.get("checkpoint_path")
    if checkpoint_path is not None:
        builder = posterior_nn(model="maf")
        net = restore_torch(build_sbi_estimator(builder, x_obs_stats, prior), checkpoint_path, device_sbi)
        posterior = inference.build_posterior(net, sample_with="mcmc").set_default_x(x_obs_stats)
        samples = posterior.sample((n_samples,), num_chains=int(kwargs.get("mcmc_chains", 20)))
        return samples.detach().cpu().numpy()

    proposal = prior

    for r in range(num_rounds):
        print(f"--- SBI Round {r+1}/{num_rounds} ---")

        if r == 0 and initial_training_data is not None:
            x_train_np, theta_train_np = initial_training_data
            n0 = min(sims_per_round, theta_train_np.shape[0])
            theta_np = theta_train_np[:n0].astype(np.float32)
            x_sim_np = x_train_np[:n0].astype(np.float32)
            theta = torch.from_numpy(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np)
        else:
            theta = proposal.sample((sims_per_round,))
            theta_np = theta.cpu().numpy()
            x_sim_np = task.simulator(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np).float()

        x = calculate_summary_statistics(x_sim_torch)

        theta = theta.to(device_sbi)
        x = x.to(device_sbi)

        train_kwargs = {
            "max_num_epochs": max_epochs,
            "validation_fraction": train_validation_fraction,
            "stop_after_epochs": train_stop_after_epochs,
            "learning_rate": train_learning_rate,
            "training_batch_size": train_batch_size,
            "show_train_summary": show_train_summary,
        }
        if isinstance(inference, NPE_A):
            final_round = r == num_rounds - 1
            train_kwargs["final_round"] = final_round

        with torch.enable_grad():
            inference = inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train(**train_kwargs)

        posterior = inference.build_posterior(
            density_estimator, sample_with="mcmc"
        ).set_default_x(x_obs_stats)
        proposal = posterior

    mcmc_chains = int(kwargs.pop("mcmc_chains", 20))
    print(f"Sampling from final posterior ({mcmc_chains} chains)...")

    samples = posterior.sample((n_samples,), num_chains=mcmc_chains)

    return samples.detach().cpu().numpy()


def train_sbi_model(
    model_type,
    task,
    device="cpu",
    num_rounds=1,
    sims_per_round=1000,
    max_epochs=1000,
    initial_training_data=None,
    x_obs=None,
    validation_fraction=0.1,
    stop_after_epochs=20,
    learning_rate=5e-4,
    training_batch_size=200,
    show_train_summary=False,
):
    """Train an SBI model and return trained artifacts for later sampling.

    This separates training from sampling so the same model can be evaluated
    on multiple different x_obs without retraining.

    For ``num_rounds > 1`` (sequential SNPE-C), each round's proposal is the
    posterior conditioned on a specific observation, so ``x_obs`` (the
    observation used for inference) must be passed; rounds 2+ sample the
    proposal via ``posterior.sample(...)`` and require ``set_default_x``.
    For amortized training evaluated on many observations (e.g.,
    DataSizeDependence), keep ``num_rounds == 1``.

    Returns
    -------
    dict with keys: inference, density_estimator, prior, device_sbi
    """
    print(
        f"=== Training SBI model (Rounds={num_rounds}, Sims/Round={sims_per_round}, "
        f"MaxEpochs={max_epochs}) ==="
    )

    # MPS has autograd gaps (linalg_qr etc.), force CPU. CUDA is fine.
    if device and "cuda" in str(device).lower():
        device_sbi = device
    elif device and "mps" in str(device).lower():
        print("  MPS detected → forcing CPU for SBI (MPS autograd not fully supported)")
        device_sbi = "cpu"
    elif torch.cuda.is_available():
        device_sbi = "cuda"
        print("  CUDA available → using GPU for SBI")
    else:
        device_sbi = "cpu"

    low = torch.tensor(task.lower, dtype=torch.float32, device=device_sbi)
    high = torch.tensor(task.upper, dtype=torch.float32, device=device_sbi)
    prior = BoxUniform(low=low, high=high, device=str(device_sbi))

    if x_obs is not None:
        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs).float()
        if x_obs.ndim == 2:
            x_obs_input = x_obs.unsqueeze(0)
        elif x_obs.ndim == 3:
            x_obs_input = x_obs
        else:
            raise ValueError(
                f"Expected x_obs shape (n, d) or (1, n, d), got {x_obs.shape}"
            )
        x_obs_stats = calculate_summary_statistics(x_obs_input)
    else:
        x_obs_stats = None

    if num_rounds > 1 and x_obs_stats is None:
        raise ValueError(
            "Multi-round SNPE-C (num_rounds>1) requires conditioning on the "
            "observation used for inference: pass x_obs so that each round's "
            "proposal is p(theta | x_obs). For amortized evaluation over many "
            "observations without retraining (e.g., DataSizeDependence), keep "
            "num_rounds=1."
        )

    model_type_lower = model_type.lower()
    if model_type_lower == "snpe_a":
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type_lower == "snpe_b":
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type_lower in ["snpe_c", "npe", "snpe"]:
        inference = NPE(prior=prior, device=device_sbi)
    else:
        print(
            f"Warning: Unknown model_type '{model_type}'. Defaulting to NPE (SNPE-C)."
        )
        inference = NPE(prior=prior, device=device_sbi)

    proposal = prior
    density_estimator = None

    for r in range(num_rounds):
        print(f"--- SBI Round {r+1}/{num_rounds} ---")

        if r == 0 and initial_training_data is not None:
            x_train_np, theta_train_np = initial_training_data
            n0 = min(sims_per_round, theta_train_np.shape[0])
            theta_np = theta_train_np[:n0].astype(np.float32)
            x_sim_np = x_train_np[:n0].astype(np.float32)
            theta = torch.from_numpy(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np)
        else:
            theta = proposal.sample((sims_per_round,))
            theta_np = theta.cpu().numpy()
            x_sim_np = task.simulator(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np).float()

        x = calculate_summary_statistics(x_sim_torch)

        theta = theta.to(device_sbi)
        x = x.to(device_sbi)

        train_kwargs = {
            "max_num_epochs": max_epochs,
            "validation_fraction": validation_fraction,
            "stop_after_epochs": stop_after_epochs,
            "learning_rate": learning_rate,
            "training_batch_size": training_batch_size,
            "show_train_summary": show_train_summary,
        }
        if isinstance(inference, NPE_A):
            final_round = r == num_rounds - 1
            train_kwargs["final_round"] = final_round

        with torch.enable_grad():
            inference = inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train(**train_kwargs)

        posterior = inference.build_posterior(
            density_estimator, sample_with="mcmc"
        )
        if x_obs_stats is not None:
            posterior = posterior.set_default_x(x_obs_stats)
        proposal = posterior

    print("SBI training complete.")
    return {
        "inference": inference,
        "density_estimator": density_estimator,
        "prior": prior,
        "device_sbi": device_sbi,
    }


def sample_from_trained_sbi(trained, x_obs, n_samples, mcmc_chains=20):
    """Sample from a pre-trained SBI model conditioned on x_obs.

    Parameters
    ----------
    trained : dict
        Output of train_sbi_model() with keys:
        inference, density_estimator, prior, device_sbi
    x_obs : np.ndarray or torch.Tensor, shape (n, d_x) or (1, n, d_x)
        The observation to condition on.
    n_samples : int
        Number of posterior samples to draw.
    mcmc_chains : int
        Number of MCMC chains for sampling.

    Returns
    -------
    np.ndarray of shape (n_samples, d)
    """
    inference = trained["inference"]
    density_estimator = trained["density_estimator"]

    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()

    if x_obs.ndim == 2:
        x_obs_input = x_obs.unsqueeze(0)
    elif x_obs.ndim == 3:
        x_obs_input = x_obs
    else:
        raise ValueError(f"Expected x_obs shape (n, d) or (1, n, d), got {x_obs.shape}")

    x_obs_stats = calculate_summary_statistics(x_obs_input)

    posterior = inference.build_posterior(
        density_estimator, sample_with="mcmc"
    ).set_default_x(x_obs_stats)

    print(f"Sampling from trained SBI posterior ({mcmc_chains} chains)...")
    samples = posterior.sample((n_samples,), num_chains=mcmc_chains)

    return samples.detach().cpu().numpy()
