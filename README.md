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
- **BayesFlow** — the KL-divergence variant of the same summary-learning
  framework (the estimator architecture follows Radev et al., 2020).

The following existing methods are used as baselines:

- **DNN-ABC** — ABC rejection sampling with a learned (regression) summary
  network.
- **SNPE-C / NPE** — neural posterior estimation, with and without a summary
  network.
- **NLE / TD-NPSE** — observation-wise likelihood and diffusion-based posterior
  estimators (`Gaussian/`).
- **NPE-RS** — robust NPE variant for misspecified simulators (`Radio/`).

## Reproducing the paper results

All experiments below reproduce specific tables and figures in the
"Numerical Experiments" section of the manuscript *"Neural Estimation of
Summary Statistics for Simulation-based Inference with Non-asymptotic
Guarantees"*. The mapping between code and paper output is:

| Task directory | Manuscript subsection | Reproduces |
| --- | --- | --- |
| `Gaussian/` | Toy Example With Multimodal Posterior | Table 1 ("Results for toy examples."): MMD and \(\sqrt{n}\times\)sW between each estimated posterior and the ground-truth posterior for \(n=25,50,100\), averaged over 10 replications. |
| `Gaussian/DataSizeDependence/` | Toy Example With Multimodal Posterior | Figure 2 ("Sliced-Wasserstein (sW) distance between the estimated and ground-truth posteriors, as a function of the observation count \(n\) under different training sizes \(N\)"): the scaling study of SMMD vs NPE, reporting \(\sqrt{n}\times\)sW for \(N=20{,}000/30{,}000/40{,}000\) and \(n=10/75/150/225/300\), averaged over 25 test datasets. |
| `SIR/` | SIR Model With Small Training Budget | Table 2 ("Results for SIR model."), the calibration-curves figure and the illustration of the approximate posterior before and after refinement. |
| `LotkaVoletrra/` | Population Ecology Model With Vague Prior | Table 3 ("Results for Lotka-Volterra model.") and the figure showing posterior density plots across the stages of the Refine+ procedure. |
| `Radio/` | Real Data Application With A Misspecified Simulator | The predictive-performance figure (amortized vs refined) and the RMSE metric reported at the end of the subsection. |

The exact output files and directories produced by each experiment (metrics
tables, posterior samples, plots and logs) are listed in the
`## Outputs` section of the corresponding task README.

More detail on each experiment:

- **`Gaussian/` — toy example with multimodal posterior.** SMMD, MMD,
  BayesFlow, DNN-ABC and SNPE-C at observation counts `n = 25, 50, 100`,
  with local refinement for SMMD / MMD / BayesFlow. NLE and TD-NPSE are
  trained once on a large single-observation dataset (50,000 pairs, uniform
  prior) by standalone scripts; the main experiment loads their checkpoints
  and only performs posterior inference and metric evaluation (no per-round
  training or local refinement).
  - `Gaussian/run_nle_singleobs_train.py` — auxiliary: trains the NLE
    model and exports `data/nle_checkpoint.pt`.
  - `Gaussian/run_td_npse_singleobs_train_pairplots.py` — auxiliary:
    TD-NPSE training and pairplot diagnostics (the `pairplots` suffix refers
    to the posterior-comparison figures it draws after training; see the
    `Gaussian/README.md` for details).
  - `Gaussian/DataSizeDependence/` — scaling study: SMMD vs NPE across
    training budgets (`N = 20,000 / 30,000 / 40,000`) and observation
    counts (`n = 10 / 75 / 150 / 225 / 300`), studying how the estimation
    error scales with the observation count under different training
    budgets.
- **`SIR/` — epidemic calibration experiment.** SMMD, MMD, BayesFlow and
  DNN-ABC under a simulation-based calibration protocol, with local
  refinement.
- **`LotkaVoletrra/` — population-ecology model with vague prior.** SMMD, MMD
  and BayesFlow (Refine+), plus the DNN-ABC and SNPE-C baselines.
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

`Gaussian/` additionally contains the standalone NLE / TD-NPSE training
scripts (`run_nle_singleobs_train.py`,
`run_td_npse_singleobs_train_pairplots.py`) and the released checkpoints
under `data/` (see "Running an experiment" below).

The `models/` subdirectory holds the network definitions used by the task:
`smmd.py`, `mmd.py`, `bayesflow_net.py` and `dnnabc.py` implement the
summary-network / posterior-estimator building blocks of the corresponding
methods, and `sbi_wrappers.py` is a thin wrapper around the `sbi` package
used to train and sample the SNPE-C / NPE methods. It appears in
`Gaussian/` (also used by the `DataSizeDependence/` sub-experiment),
`LotkaVoletrra/` and `Radio/` (where it additionally implements the NPE-RS
baseline); `SIR/` does not use SNPE-C/NPE and contains no wrapper.

## Installation

The experiments run with **Python 3.13**. All package versions are pinned in
`environment/environment.yml` (Conda) and `environment/requirements.txt`
(pip); the key packages are **PyTorch 2.10**, **sbi 0.26.1** (SNPE-C / NPE),
**BayesFlow 2.0.8**, **pyABC 0.12.16** and **POT 0.9.6** (sliced-Wasserstein
metric). Software central to the methods is cited in the manuscript's
References section (e.g., BayesFlow); see also `environment/README.md`.

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

The pretrained NLE and TD-NPSE checkpoints are included in the repository
under `Gaussian/data/`, so the main experiment runs out of the box.  To
retrain them from scratch instead (only required once per dataset):

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
| `num_rounds` / `n_rounds` | replications (Gaussian, LotkaVoletrra) or SBI rounds (SNPE-C/NPE) |
| `n_observation` / `n_observation_list` | observations per dataset (Gaussian; paper values 25/50/100) |

## Implementation notes

We control variability as much as possible: every experiment fixes its
random seed (`seed` in `config.json`), data generation and training are
deterministic given the seed, and CUDA work is configured for
deterministic algorithms where supported. Nevertheless, the performance of
an amortized estimator can still vary somewhat when it is trained on a
different machine (GPU, CUDA version, or framework build), because
floating-point reductions and non-deterministic kernels differ across
hardware.

For this reason, the local ABC-MCMC refinement step may need mild
re-tuning to make sure the chains converge to the correct
summary-conditioned posterior. The main knob is the proposal step-size
coefficient (`proposal_std` in each task's `refinement_config` /
`refine_config`, `proposal_std_coeff` in LotkaVoletrra); values in the
range **0.2–0.6** are a good starting point. If the acceptance rate is very
low, decrease the coefficient; if the chains barely move from the amortized
samples, increase it.
