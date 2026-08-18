# Summary Learning for Simulation-Based Inference

Code accompanying the manuscript *"Neural Estimation of Summary Statistics for
Simulation-based Inference with Non-asymptotic Guarantees"*.

This repository contains the benchmark experiments comparing several
summary-learning / simulation-based inference (SBI) methods on four tasks:

| Task | Directory | Parameter dim | Observation type |
| --- | --- | --- | --- |
| Gaussian (stereographic) | `Gaussian/` | 5 | deep-set observations |
| Lotka–Volterra | `LotkaVoletrra/` | 5 | noisy time series |
| Radio (Turin channel) | `Radio/` | 4 | impulse-response realizations |
| SIR epidemic | `SIR/` | 2 | infection-curve time series |

## Methods

The proposed methods are:

- **SMMD** — summary network + conditional generator trained with sliced
  maximum mean discrepancy.
- **MMD** — the same architecture trained with multi-scale MMD (V-statistic).

The following existing methods are used as baselines:

- **BayesFlow** — amortized posterior estimation with a summary network
  (Radev et al., 2020).
- **DNN-ABC** — ABC rejection sampling with a learned (regression) summary
  network.
- **SNPE-C / NPE** — neural posterior estimation, with and without a summary
  network.
- **NLE / TD-NPSE** — observation-wise likelihood and diffusion-based posterior
  estimators (`Gaussian/`).
- **NPE-RS** — robust NPE variant for misspecified simulators (`Radio/`).

## Experiments

Each task directory contains a main experiment and, in some cases, auxiliary
sub-experiments:

- **`Gaussian/` — toy example with multimodal posterior.** SMMD, MMD,
  BayesFlow, DNN-ABC and SNPE-C at observation counts `n = 25, 50, 100`,
  with local refinement. NLE and TD-NPSE are trained once on a large
  single-observation dataset (50,000 pairs, uniform prior) by standalone
  scripts; the main experiment loads their checkpoints and only performs
  posterior inference and metric evaluation (no per-round training or
  local refinement).
  - `Gaussian/run_nle_singleobs_train.py` — auxiliary: trains the NLE
    model and exports `data/nle_checkpoint.pt`.
  - `Gaussian/run_td_npse_singleobs_train_pairplots.py` — auxiliary:
    TD-NPSE training and pairplot diagnostics.
  - `Gaussian/DataSizeDependence/` — scaling study: SMMD vs NPE across
    training budgets (`N = 20,000 / 30,000 / 40,000`) and observation
    counts (`n = 10 / 75 / 150 / 225 / 300`).
- **`SIR/` — epidemic calibration experiment.** SMMD, MMD, BayesFlow and
  DNN-ABC under a simulation-based calibration protocol, with local
  refinement.
- **`LotkaVoletrra/` — population-ecology model with vague prior.** SMMD, MMD,
  BayesFlow, DNN-ABC and SNPE-C with the Refine+ refinement variant.
- **`Radio/` — real data with a misspecified simulator.** SMMD, MMD,
  BayesFlow, DNN-ABC and NPE-RS.

## Repository layout

```
.
├── sbi_common/            # Shared losses, networks and ABC helpers
├── Gaussian/              # Gaussian task
│   └── DataSizeDependence/  # scaling study (SMMD vs NPE)
├── LotkaVoletrra/         # Lotka-Volterra task
├── Radio/                 # Radio (Turin) task
├── SIR/                   # SIR task
├── environment/           # Conda / pip environment files
├── tests/                 # Numerical equivalence and smoke tests
└── pyproject.toml
```

Each task directory contains:

- `data_generation.py` — simulator, prior and task class;
- `run_experiment.py` — training, refinement and metric evaluation entry point;
- `config.json` — experiment configuration (dataset size, epochs, model hyper-parameters);
- `models/` — per-task network definitions (thin wrappers over `sbi_common` where shared);
- `utilities.py` — task-specific MCMC refinement and evaluation helpers;

## Installation

### Option 1 — Conda

```bash
conda env create -f environment/environment.yml
conda activate sbi-summary
```

### Option 2 — pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
```

On NVIDIA GPUs, install the matching CUDA build of PyTorch before the other
packages (the experiments were run on Linux with CUDA 12.8).

## Running an experiment

From the repository root:

```bash
python -m Gaussian.run_experiment                     # default config
python -m SIR.run_experiment --config SIR/config.json
```

or from inside a task directory (equivalent):

```bash
cd Gaussian && python run_experiment.py
cd SIR && python run_experiment.py --config config.json
```

Sub-experiments are run in the same way:

```bash
python Gaussian/DataSizeDependence/run_experiment.py --config Gaussian/DataSizeDependence/config.json
```

Before running the Gaussian main experiment, train the NLE and TD-NPSE
checkpoints (only required once per dataset):

```bash
python Gaussian/run_nle_singleobs_train.py --n-train 50000 --epochs 500
python Gaussian/run_td_npse_singleobs_train_pairplots.py --n-train 50000
```

The Gaussian main experiment then reloads these checkpoints (config entries
`models_config.nle` / `models_config.td_npse` with `eval_mode:
"reload_checkpoint"`) and only performs posterior inference and metric
evaluation. Note the following:

- `run_nle_singleobs_train.py` saves its checkpoint directly to
  `Gaussian/data/nle_checkpoint.pt` (the default `--checkpoint-path`).
- `run_td_npse_singleobs_train_pairplots.py` writes its checkpoint under
  `Gaussian/results/td_npse_singleobs_train_pairplots/<exp_name>/artifacts/td_npse_checkpoint.pt`,
  where `<exp_name>` encodes `n_train`, `train_n_obs`, `epochs` and `seed`
  (e.g. `ntrain50000_trainobs1_e5000_seed0`). Copy it to the location
  expected by the main experiment before running it:

  ```bash
  cp Gaussian/results/td_npse_singleobs_train_pairplots/ntrain50000_trainobs1_e5000_seed0/artifacts/td_npse_checkpoint.pt \
     Gaussian/data/td_npse_checkpoint.pt
  ```

- In reload mode, `run_experiment.py` rebuilds both models from the saved
  checkpoint (`model_kwargs` plus state dict), so their training and
  inference hyperparameters (e.g. NLE epochs, TD-NPSE diffusion/sampling
  settings) come from the training run, not from `config.json`; the
  training-related `nle`/`td_npse` config fields are therefore ignored by
  the main experiment. Set training hyperparameters through the standalone
  scripts' CLI arguments (defaults: NLE 500 epochs; TD-NPSE up to 5000
  epochs with early stopping).
- To confirm the checkpoints were loaded, look for
  `Reloading NLE checkpoint for evaluation: ...` and
  `Reloading TD-NPSE checkpoint for evaluation: ...` in the main experiment
  log. A missing checkpoint raises `FileNotFoundError`.

The scripts insert the task directory and the repository root into
`sys.path`, so they work from any working directory. Results are written to
`results*/` subdirectories next to each task (git-ignored; regenerate with the
same `config.json` to reproduce).

## Configuration

Each task has its own `config.json` (there is deliberately no shared schema:
the tasks expose different hyper-parameters). Common keys:

| Key | Meaning |
| --- | --- |
| `dataset_size` | number of training pairs `(theta, x)` |
| `batch_size` / `learning_rate` | training hyper-parameters |
| `models_to_run` | list of methods to evaluate |
| `models_config.<model>.*` | per-model hyper-parameters |
| `n_samples_posterior` | posterior samples per method |
| `num_rounds` / `n_rounds` | training rounds (SMMD/MMD) or SBI rounds |
| `n_observation` / `n_observation_list` | observations per dataset (Gaussian; paper values 25/50/100) |
