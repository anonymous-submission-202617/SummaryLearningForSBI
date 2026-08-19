# Gaussian (stereographic) toy example

Five-parameter bivariate Gaussian model with a nonlinear (stereographic)
transformation; the posterior is multimodal.  The experiment compares the
proposed methods (SMMD, MMD) with BayesFlow, DNN-ABC, SNPE-C (NPE), NLE and
TD-NPSE at observation counts `n = 25, 50, 100`, each with 10 replications and
local refinement.

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

Each `n` uses the same `config.json` with two changes:

| `n` | `n_observation` / `n_observation_list` | `refine_config.nsims` |
| --- | --- | --- |
| 25 | `25` / `[25]` | `4` |
| 50 | `50` / `[50]` | `2` |
| 100 | `100` / `[100]` | `1` |

The refinement likelihood is estimated from `nsims` simulator calls per MCMC
proposal; we use `nsims = 100 / n` (i.e. 4, 2, 1 for `n = 25, 50, 100`), which
is already reflected in the table above.

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
cp config.json config_n50.json   # edit n_observation=50, nsims=2
tmux new-session -d -s gauss_n50 \
  "python run_experiment.py --config config_n50.json 2>&1 | tee run_n50.log"

# n = 100
cp config.json config_n100.json  # edit n_observation=100, nsims=1
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
- NLE and TD-NPSE are observation-wise estimators; the main experiment loads
  their pretrained checkpoints from `data/` (`nle_checkpoint.pt`,
  `td_npse_checkpoint.pt`, included in this repository) and only performs
  posterior inference and metric evaluation. We release the exact model
  parameters used for the experiments in the paper, so the NLE and TD-NPSE
  entries in the tables can be reproduced directly without retraining. The
  checkpoints were generated once by the standalone training scripts
  (`run_nle_singleobs_train.py` and `run_td_npse_singleobs_train_pairplots.py`).

## Outputs

Results are written to `results_25/` (or `results_50/`, `results_100/`):
`experiment_results.csv`, per-round posterior plots, and posterior samples.
