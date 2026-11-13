# Lotka-Volterra model

Population-ecology benchmark (prey--predator dynamics generated with the
Gillespie algorithm) under a vague prior, where the proposed methods use the
**Refine+** variant (sequential training followed by ABC-MCMC local
refinement).

## Paper result reproduced

This experiment reproduces the **Lotka-Volterra results of the manuscript's
"Numerical Experiments" section** (subsection *Population Ecology Model With
Vague Prior*):

- **Table 3 ("Results for Lotka-Volterra model.")** — bias, credible-interval
  length, coverage and conditional calibration error, together with average
  runtime. See the remark below for how we average BayesFlow results;
- the **Refine+ figure** — posterior density plots across the stages of the
  Refine+ procedure ("Amortized", "Sequential" and "Refine+ (MCMC)") for the
  SMMD variant. It is produced by `plot_refinement_comparison` and written
  to `results/models/smmd/round_<r>/comparison/smmd_refinement_comparison.png`
  (the code labels the last stage "RefinePlus"); a combined
  all-methods version is saved as
  `results/comparisons/round_<r>/all_models_refinement_comparison.png`.
  These comparison figures are expensive to render and are **disabled by
  default** (see "Runtime" below); set
  `plot_config.save_refinement_comparison_plots` to `true` in `config.json`
  to regenerate them.

> **Remark (LV BayesFlow).** The KL loss can cause exploding gradients and
> make training fail. We average only runs that finish and yield usable
> posterior samples, even if the loss fluctuates or starts decreasing late.

All methods are compared under a matched simulation budget (total number of
simulator calls across training and refinement). SNPE-C uses the standard
hand-crafted 9-dimensional summary vector; SMMD, MMD, BayesFlow and DNN-ABC
learn their summaries.

## Settings

From `config.json`:

- `dataset_size = 25,600` training pairs; `n_time_steps = 151`, `dt = 0.2`;
- `num_rounds = 100` replications at a fixed true parameter;
- `n_samples_posterior = 1,000`;
- models: `smmd`, `mmd`, `bayesflow` (Refine+), `dnnabc`, `snpe_c`.

## Runtime

This experiment is computationally heavy: a full run (all models with the
Refine+ procedure and 100 replications) takes on the order of a week or
more on a single GPU. Rendering the refinement-comparison figures is one of
the most expensive parts, so it is turned **off by default** in the shipped
`config.json` (`plot_config.save_refinement_comparison_plots: false`); set
it to `true` only when the paper's Refine+ figure needs to be regenerated.

## Running

```bash
python run_experiment.py --config config.json
```

## Outputs

Written to `results/`:

- `tables/<model>_results.csv` — per-replication and aggregated metrics
  behind Table 3;
- `calibration/calibration_summary.csv` — the calibration-error summary
  behind the conditional calibration error reported in Table 3
  (per-model/per-stage `.npz` files hold the full diagnostics);
- per-round posterior plots and the combined Refine+ stage figures
  (`*refineplus*`, `final_methods_logtheta.png` / `.theta`) used in the
  Refine+ figure.

## Execution order

The experiment is driven by a single entry point that trains the amortized
models, runs the Refine+ sequential training and ABC-MCMC refinement,
evaluates the 100 replications, and writes all tables and figures:

```bash
python run_experiment.py --config config.json
```

There is no separate training or plotting script to run beforehand.

## Implementation notes

Every run fixes its random seed (`seed` in `config.json`) and, given the
seed, data generation and training are deterministic. Nevertheless, the
performance of an amortized estimator can still vary somewhat when it is
trained on a different machine (GPU, CUDA version, or framework build).
The ABC-MCMC refinement step may therefore need mild re-tuning so that the
chains converge to the correct summary-conditioned posterior; the main knob
is the proposal step-size coefficient
(`refinement_config.proposal_std_coeff`), with values in the range
**0.2–0.6** as a good starting point.
