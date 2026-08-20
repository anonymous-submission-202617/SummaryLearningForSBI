# Gaussian (stereographic) toy example

Five-parameter bivariate Gaussian model with a nonlinear (stereographic)
transformation; the posterior is multimodal.  The experiment compares the
proposed methods (SMMD, MMD) with BayesFlow, DNN-ABC, SNPE-C, NLE and
TD-NPSE at observation counts `n = 25, 50, 100`, each with 10 replications and
local refinement.

## Paper result reproduced

This experiment reproduces **Table 1 ("Results for toy examples.")** in the
subsection *Toy Example With Multimodal Posterior* of the manuscript's
"Numerical Experiments" section. For each observation count `n`, the table
reports the MMD and the sliced Wasserstein distance (sW, rescaled as
\(\sqrt{n}\times\)sW) between each estimated posterior and the ground-truth
posterior, averaged over the 10 replications. The code produces the
amortized and refined columns for SMMD, MMD and BayesFlow, and the
corresponding entries for the DNN-ABC, SNPE-C, NLE and TD-NPSE baselines.

The sub-experiment [`DataSizeDependence/`](DataSizeDependence/README.md)
reproduces the scaling figure (sW vs observation count under different
training budgets) of the same manuscript subsection.

The default configuration (`config.json`) runs `n = 25` with the exact
hyper-parameters used for the results in the paper.  The same hyper-parameters
are used for `n = 50, 100`; only the two fields listed below change.

All experiments use a fixed uniform prior `Unif([-3, 3])^5` on the parameters;
the prior is hard-coded in `data_generation.py`, so there is no `prior_type`
option to configure.

## Quick check (smoke test)

```bash
python run_experiment.py --config config_smoke.json
```

`config_smoke.json` uses a tiny dataset, a single round and a short MCMC chain
so the full pipeline (training, true-posterior sampling, refinement, metrics,
plots) completes in a few minutes.

## Reproducing the paper runs

Each `n` uses the same `config.json` with one change:

| `n` | `n_observation` / `n_observation_list` |
| --- | --- |
| 25 | `25` / `[25]` |
| 50 | `50` / `[50]` |
| 100 | `100` / `[100]` |

All runs use `seed = 0`; within a run the ten replications use round seeds
`seed + (round - 1)`, i.e. `0..9`.  Since `n = 25, 50, 100` are launched as
independent processes, sharing the same seed does not couple the runs.

## Running the three sample sizes in parallel

The runs are independent, so on a multi-core server they can be launched as
three `tmux` sessions (one per sample size):

```bash
# n = 25 (default config)
tmux new-session -d -s gauss_n25 \
  "python run_experiment.py --config config.json 2>&1 | tee run_n25.log"

# n = 50
cp config.json config_n50.json   # edit n_observation=50
tmux new-session -d -s gauss_n50 \
  "python run_experiment.py --config config_n50.json 2>&1 | tee run_n50.log"

# n = 100
cp config.json config_n100.json  # edit n_observation=100
tmux new-session -d -s gauss_n100 \
  "python run_experiment.py --config config_n100.json 2>&1 | tee run_n100.log"
```

Attach with `tmux attach -t gauss_n25`, list with `tmux ls`, and kill with
`tmux kill-session -t gauss_n25`.

## True posterior sampling (MCMC)

The reference posterior is sampled with the same random-walk Metropolis (RWM)
sampler used by the refinement step (`run_gaussian_posterior_mcmc` in
`utilities.py`).  `true_posterior_config` sets `n_draws = 1`, `n_tune_chain =
1000`, `chains = 2000` and `proposal_scale_coeff = 0.55`, so the proposal
scale is `0.55 / sqrt(n)`.  Each of the 2000 independent chains contributes
one draw after 1000 burn-in steps, yielding 2000 reference samples per
observation.

## Models and checkpoints

- SMMD / MMD / BayesFlow / DNN-ABC / SNPE-C are trained inside the experiment
  (`models_config`).
- SNPE-C / NPE are trained and sampled through `models/sbi_wrappers.py`, a
  thin wrapper around the `sbi` package; the `DataSizeDependence/`
  sub-experiment reuses the same wrapper for its NPE runs.
- NLE and TD-NPSE are observation-wise estimators; the main experiment loads
  their pretrained checkpoints from `data/` (`nle_checkpoint.pt`,
  `td_npse_checkpoint.pt`, included in this repository) and only performs
  posterior inference and metric evaluation. We release the exact model
  parameters used for the experiments in the paper, so the NLE and TD-NPSE
  entries in the tables can be reproduced directly without retraining. The
  checkpoints were generated once by the standalone training scripts
  (`run_nle_singleobs_train.py` and `run_td_npse_singleobs_train_pairplots.py`):
  - `run_nle_singleobs_train.py` — trains the NLE model and exports
    `data/nle_checkpoint.pt`;
  - `run_td_npse_singleobs_train_pairplots.py` — trains the TD-NPSE model
    and, as the `pairplots` suffix indicates, additionally evaluates the
    fitted estimator: for each observation count it draws the MCMC true
    posterior at the fixed true parameter, samples the TD-NPSE posterior,
    and writes pairplot / marginal comparison figures and MMD / sW metric
    tables to `results/td_npse_singleobs_train_pairplots/<exp_name>/`, where
    `<exp_name>` encodes the training configuration (e.g.
    `ntrain50000_trainobs1_e5000_seed0`). The TD-NPSE checkpoint to copy to
    `data/` is
    `results/td_npse_singleobs_train_pairplots/<exp_name>/artifacts/td_npse_checkpoint.pt`.

## Outputs

Each run writes to `results_<n>/` (`results_25/`, `results_50/` or
`results_100/`):

- `experiment_results.csv` — per-round, per-model MMD and
  \(\sqrt{n}\times\)sW values (amortized and refined) behind Table 1;
- `final_summary.csv` — per-model summary statistics (mean / median over
  the 10 replications);
- `posterior_samples/true_reference/round_<r>/samples.npy` — the MCMC
  ground-truth posterior samples for each replication;
- `plots/round_<r>/true_posterior/posterior.png` — the ground-truth
  posterior; `plots/round_<r>/<model>/` — per-model `amortized_posterior.png`,
  `refined_posterior.png` (SMMD / MMD / BayesFlow) and
  `<model>_refinement_comparison.png`;
- `plots/round_<r>/combined_posteriors.png` and
  `plots/round_<r>/all_models_refinement_comparison.png` — all-model
  marginal-posterior figures;
- `plots/round_<r>/loss/` — training loss curves (`*.png` with matching
  `*.csv` data);
- `terminal.log` — full run log.

## Execution order

1. *(Optional)* Train the observation-wise baselines once, if you do not want
   to use the released checkpoints in `data/`:
   `run_nle_singleobs_train.py`, then
   `run_td_npse_singleobs_train_pairplots.py` (copy the TD-NPSE checkpoint to
   `data/td_npse_checkpoint.pt` as described above).
2. Run the main experiment for each observation count:
   `python run_experiment.py --config config.json` with
   `n_observation = 25, 50, 100` (three independent runs, see "Running the
   three sample sizes in parallel").
3. *(Optional, independent)* Run the scaling study:
   `cd DataSizeDependence && python run_experiment.py --config config.json`.

## Implementation notes

Every run fixes its random seed (`seed` in `config.json`) and, given the
seed, data generation and training are deterministic. Nevertheless, the
performance of an amortized estimator can still vary somewhat when it is
trained on a different machine (GPU, CUDA version, or framework build).
The ABC-MCMC refinement step may therefore need mild re-tuning so that the
chains converge to the correct summary-conditioned posterior; the main knob
is the proposal step-size coefficient (`refine_config.proposal_std`),
with values in the range **0.2–0.6** as a good starting point.
