# Data-size dependence (Gaussian task)

Scaling study of the Gaussian toy example: how the posterior error evolves
with the observation count `n` under different training budgets `N`,
comparing **SMMD** with **NPE** (SNPE-C without sequential training).

## Paper result reproduced

This experiment reproduces **Figure 2** of the manuscript ("Sliced-Wasserstein
(sW) distance between the estimated and ground-truth posteriors, as a
function of the observation count \(n\) under different training sizes
\(N\)"), which appears in the subsection *Toy Example With Multimodal
Posterior*. The figure reports the rescaled distance \(\sqrt{n}\times\)sW,
which compares posteriors on the root-\(n\) local scale, as lines with shaded
regions showing the mean and standard deviation over the 25 test datasets.

## Settings

From `config.json`:

- observation counts `n_obs_list`: `10 / 75 / 150 / 225 / 300`;
- training budgets `train_sizes`: `20,000 / 30,000 / 40,000`;
- `n_try = 25` test datasets per `(n, N)` combination;
- `seed = 42`.

Each trial uses the fixed true parameter `TRUE_PARAMS` of the Gaussian task
(the same value used by the main Gaussian experiment) and draws a fresh
observation from the simulator. The metric is the sliced Wasserstein
distance between the estimated posterior samples and ground-truth posterior
samples (2000 each), rescaled by \(\sqrt{n}\).

## Running

From this directory:

```bash
python run_experiment.py --config config.json
```

## Implementation notes

The run uses a fixed seed (`seed = 42`) and deterministic data generation
and training given the seed. Since the performance of the trained
amortized estimators can still vary somewhat across machines (GPU, CUDA
version, or framework build), the reported \(\sqrt{n}\times\)sW values may
differ slightly from the manuscript; retraining with the same `config.json`
reproduces the figure qualitatively. This sub-experiment evaluates
amortized posteriors only and involves no ABC-MCMC refinement step.

or from the repository root:

```bash
python Gaussian/DataSizeDependence/run_experiment.py --config Gaussian/DataSizeDependence/config.json
```

## Outputs

Written to `results_datasize_dependence/`:

- `results_raw.csv` / `results_agg.csv` — per-trial and aggregated
  \(\sqrt{n}\times\)sW values;
- `datasize_dependence_swd_transposed.png` — the transposed multi-panel
  figure (one panel per training budget, `n` on the x-axis) used in the
  manuscript.

## Execution order

This sub-experiment is self-contained: it trains SMMD and NPE from scratch
for every `(n, N)` combination, so it has no dependency on the main Gaussian
experiment and can be run at any time with a single command:

```bash
python run_experiment.py --config config.json
```
