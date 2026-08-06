"""ABC with Wasserstein-2 / learned-summary distances via pyABC.

Used by the Gaussian and Lotka-Volterra experiments (identical code in both,
kept here as a single shared implementation).
"""

import os
import logging

import numpy as np
import torch
import pyabc
import ot

# Set pyabc logging to avoid clutter.
logging.getLogger("pyabc").setLevel(logging.WARNING)


class W2Distance(pyabc.Distance):
    """Wasserstein-2 distance (via POT) between two empirical distributions."""

    def __call__(self, x, y, t=None, par=None):
        # x, y are dicts {'data': np.array (n, d_x)}.
        X = np.asarray(x["data"])
        Y = np.asarray(y["data"])

        n, m = X.shape[0], Y.shape[0]
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m

        # Squared-Euclidean cost matrix; emd2 with sqeuclidean metric is W2^2.
        M = ot.dist(X, Y, metric="sqeuclidean")
        w2_sq = ot.emd2(a, b, M)
        return np.sqrt(w2_sq)


class SummaryDistance(pyabc.Distance):
    """Euclidean distance on learned summary statistics: d(x, y) = ||S(x) - S(y)||_2."""

    def __init__(self, summary_network, device="cpu"):
        self.summary_network = summary_network
        self.device = device

    def __call__(self, x, y, t=None, par=None):
        # x, y are dicts {'data': np.array (n, d_x)}.
        X = x["data"]
        Y = y["data"]

        def get_summary(data):
            # data: (n, d_x)
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            if hasattr(self.summary_network, "predict"):  # Keras/TF (BayesFlow)
                data_np = data_tensor.cpu().numpy()
                s = self.summary_network(data_np)
                if hasattr(s, "cpu"):
                    s = s.cpu()
                if hasattr(s, "numpy"):
                    s = s.numpy()
                return torch.tensor(s).squeeze(0)  # (summary_dim,)
            with torch.no_grad():  # Torch
                s = self.summary_network(data_tensor)
            return s.cpu().squeeze(0)  # (summary_dim,)

        sx = get_summary(X)
        sy = get_summary(Y)
        return torch.norm(sx - sy).item()


def run_smc_abc(
    task,
    x_obs,
    n_samples=1000,
    max_populations=10,
    result_dir="results",
    distance_metric="w2",
    summary_network=None,
    device="cpu",
):
    """Run ABC-SMC with pyABC.

    Args:
        task: Task object exposing ``d``, ``lower``, ``upper`` and
            ``simulator(theta)``.
        x_obs: Observation ``(n, d_x)`` (or ``(1, n, d_x)``).
        n_samples: Population size / number of posterior samples.
        max_populations: Maximum number of SMC generations.
        result_dir: Directory for the temporary pyABC database.
        distance_metric: ``"w2"`` or ``"summary"``.
        summary_network: Required when ``distance_metric="summary"``.
        device: Torch device for the summary network.
    """
    metric_name = distance_metric.upper()
    print(f"\n=== Running {metric_name}-ABC (Population={n_samples}, Max Gen={max_populations}) ===")

    # 1. Uniform prior over [lower, upper] per parameter.
    prior_dict = {}
    for i in range(task.d):
        loc = task.lower[i]
        scale = task.upper[i] - task.lower[i]
        prior_dict[f"theta{i}"] = pyabc.RV("uniform", loc, scale)
    prior = pyabc.Distribution(**prior_dict)

    # 2. Model wrapper.
    def model_wrapper(params):
        theta = np.zeros(task.d)
        for i in range(task.d):
            theta[i] = params[f"theta{i}"]
        x_sim = task.simulator(theta[np.newaxis, :])[0]
        return {"data": x_sim}

    # 3. Distance.
    if distance_metric == "w2":
        distance = W2Distance()
    elif distance_metric == "summary":
        if summary_network is None:
            raise ValueError("summary_network must be provided for summary distance")
        distance = SummaryDistance(summary_network, device)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # 4. Setup ABC-SMC.
    abc = pyabc.ABCSMC(
        models=model_wrapper,
        parameter_priors=prior,
        distance_function=distance,
        population_size=n_samples,
        sampler=pyabc.sampler.SingleCoreSampler(),
    )

    # 5. Prepare observation.
    x_obs_data = x_obs[0] if x_obs.ndim == 3 else x_obs

    # Database (sqlite; removed before each run).
    os.makedirs(result_dir, exist_ok=True)
    db_path = os.path.join(result_dir, f"{distance_metric}abc_temp.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db_url = "sqlite:///" + db_path

    abc.new(db_url, {"data": x_obs_data})

    # 6. Run.
    history = abc.run(max_nr_populations=max_populations)

    # 7. Extract samples.
    df, w = history.get_distribution(m=0, t=history.max_t)

    samples = np.zeros((len(df), task.d))
    for i in range(task.d):
        samples[:, i] = df[f"theta{i}"].values

    print(f"Resampling {metric_name}-ABC posterior...")
    indices = np.random.choice(len(df), size=n_samples, p=w / w.sum(), replace=True)
    final_samples = samples[indices]

    print(f"{metric_name}-ABC Completed. Max t={history.max_t}")
    return final_samples


def run_w2abc(task, x_obs, n_samples=1000, max_populations=10, result_dir="results"):
    """Backward-compatible wrapper around :func:`run_smc_abc` with W2 distance."""
    return run_smc_abc(
        task, x_obs, n_samples, max_populations, result_dir, distance_metric="w2"
    )
