# Radio propagation (Turin channel model)

Real-data application with a misspecified simulator. Posterior inference is
performed on real radio-propagation measurements (impulse-response
realizations with `d_X = 801` frequency bins and `n = 100` observations)
using the Turin stochastic channel model as simulator, with an
uninformative prior.

## Data

The real measurements used for inference (`data/turin_obs.npy`, `n = 100`
impulse-response realizations over `d_X = 801` frequency bins) come from
the radio-propagation benchmark of Huang et al. (2023), publicly available
at <https://github.com/huangdaolang/robust-sbi>. They are used as the fixed
observed dataset throughout the experiment.

Huang, D., Bharti, A., Souza, A., Acerbi, L., & Kaski, S. (2023). Learning
Robust Statistics for Simulation-based Inference under Model
Misspecification. *Advances in Neural Information Processing Systems*, 36,
7289–7310.

## Paper result reproduced

This experiment reproduces the **Radio results of the manuscript's "Numerical
Experiments" section** (subsection *Real Data Application With A Misspecified
Simulator*):

- the **predictive-performance figure** — posterior predictive distributions
  with and without ABC-MCMC refinement (the "Amortized" and "Refined"
  panels);
- the **RMSE metric** reported at the end of the subsection: the root mean
  squared error between the posterior-predictive mean series and the
  observed mean series (over the 801 frequency bins), averaged over the 20
  replications.

## Protocol

For each of the `n_rounds = 20` replications, the script draws posterior
samples from each fitted estimator, simulates a channel power-spectrum
series for every draw, and computes the RMSE between the mean simulated
series and the observed series:

\[
\mathrm{RMSE} = \sqrt{\frac{1}{d_X}\sum_{j=1}^{d_X}
\left(\overline{y}_j - \overline{y}^{\,\mathrm{obs}}_j\right)^2},
\quad \overline{y} = \frac{1}{n_{\mathrm{pred}}}\sum_{i=1}^{n_{\mathrm{pred}}} y_i,
\]

where \(y_i\) is the simulated series for one posterior draw and
\(\overline{y}^{\,\mathrm{obs}}\) is the mean of the observed realizations.

Models: SMMD, BayesFlow, DNN-ABC and NPE-RS (see `config.json`).

## Running

```bash
python run_experiment.py --config config.json
```

## Outputs

Written to `results/`:

- `round_XX/compare_predictive_mean_std.png` and
  `compare_predictive_mean_std_refined.png` — the predictive-performance
  figures (amortized and refined) behind the paper figure;
- `round_XX/<model>_ppc.png` — per-model posterior predictive plots;
- `rmse_all_rounds.csv` — per-round RMSE values for every model;
- `rmse_summary_stats.csv` — RMSE mean / median / std over the 20 rounds
  behind the numbers reported in the manuscript.

## Execution order

1. *(Optional, sanity check)* Visualize the simulator against the real data:
   `python benchmark_data_generation.py --config config.json`.
2. Run the main experiment (it loads the real measurements from
   `data/turin_obs.npy`):

   ```bash
   python run_experiment.py --config config.json
   ```

   The script trains the amortized models, applies ABC-MCMC refinement,
   simulates posterior-predictive series, computes the per-round RMSE, and
   writes the tables and figures. No other script needs to be run first.

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
