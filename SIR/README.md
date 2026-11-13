# SIR epidemic model

Expected-coverage / calibration experiment for the susceptible--infected--
recovered (SIR) benchmark, run with a deliberately small training budget
(`dataset_size = 768`).

## Paper result reproduced

This experiment reproduces the **SIR results of the manuscript's "Numerical
Experiments" section** (subsection *SIR Model With Small Training Budget*):

- **Table 2 ("Results for SIR model.")** — quantitative calibration metrics
  aggregated over the 500 calibration replications;
- the **calibration-curves figure** — empirical coverage minus nominal
  coverage vs the nominal credibility level;
- the **refinement-illustration figure** — marginal posteriors from a single
  replication, with and without ABC-MCMC refinement.

## Protocol

The script trains each amortized model once, then evaluates prior-averaged
calibration under the prior-predictive sampling protocol underlying
simulation-based calibration (SBC): it draws `num_calibration_pairs = 500`
independent pairs \((\theta_i, X_i)\) with \(\theta_i\) from the prior and
\(X_i\) from the simulator, draws posterior samples
\(q(\theta \mid X_i)\) for each pair, and checks whether the true
\(\theta_i\) falls inside equal-tailed \(\alpha\)-credible intervals.
Coverage is aggregated over the 500 pairs; the configurable
`round_size = 100` only controls how many pairs are processed per resume
round.

Models: SMMD, MMD, BayesFlow and DNN-ABC (see `config.json`).

## Running

```bash
python run_experiment.py --config config.json
```

BayesFlow evaluation loads the summary (`*_summary.weights.h5`) and flow
(`*_inference.pt`) separately. The flow file also stores the fitted
standardization statistics. Set `save_model_params` to `true` to save
the same pair under `checkpoints/` in the calibration output directory
when training a new model.

## Outputs

Written to `results/calibration_expected_coverage/`:

- `calibration_summary.csv` / `calibration_summary_by_round.csv` — the
  aggregated calibration metrics behind Table 2;
- `calibration_curve_data.csv` — coverage-curve data behind the calibration
  figure;
- `calibration_error_curves.png` (and per-parameter variants) — the
  calibration-curves figure;
- `round_checks/round_*/` — per-replication posterior samples
  (`posterior_samples.npz`), metrics (`posterior_metrics.csv`) and marginal
  posterior comparison plots used by the refinement-illustration figure.

## Execution order

The experiment is driven by a single entry point that trains the amortized
models, samples the 500 calibration pairs, evaluates posterior coverage
with and without refinement, and writes all tables and figures:

```bash
python run_experiment.py --config config.json
```

There is no separate training or plotting script to run beforehand.

For a quick end-to-end sanity check, run the smoke configuration first:

```bash
python run_experiment.py --config config_smoke.json
```

## Implementation notes

Every run fixes its random seed (`seed` in `config.json`) and, given the
seed, data generation and training are deterministic. Nevertheless, the
performance of an amortized estimator can still vary somewhat when it is
trained on a different machine (GPU, CUDA version, or framework build).
The ABC-MCMC refinement step may therefore need mild re-tuning so that the
chains converge to the correct summary-conditioned posterior; the main knob
is the proposal step-size coefficient
(`refinement_config.proposal_std`), with values in the range **0.2–0.6**
as a good starting point.
