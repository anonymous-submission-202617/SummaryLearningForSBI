# Model weights and evaluation data

This directory contains the checkpoints and data used in evaluation mode
for the four main experiments.

| Experiment | Directory | Contents |
| --- | --- | --- |
| Gaussian | [`gaussian_weights/`](gaussian_weights/) | Checkpoints, observations, and reference posterior samples for `n = 25, 50, 100`, plus the shared NLE and TD-NPSE checkpoints. |
| SIR | [`sir_weights/`](sir_weights/) | One trained model per method, reused across the calibration pairs. |
| Lotka–Volterra | [`lv_weights/`](lv_weights/) | Checkpoints organized by method and seed. |
| Radio | [`radio_weights/`](radio_weights/) | Checkpoints organized by method and seed, plus the observed Turin channel data. |

See [Run the experiments](../README.md#run-the-experiments) for instructions
on loading these files.

> **Remark (Gaussian NLE and TD-NPSE).** We compare how these methods'
> posterior inference scales with the number of observations `n` without
> retraining. We train each method once on a larger dataset of 50,000
> simulated pairs, with one observation per parameter draw, and reuse
> [one checkpoint per method](gaussian_weights/nle_and_td_npse/) across
> all runs at `n = 25, 50, 100`.

> **Remark (LV BayesFlow).** The KL loss can cause exploding gradients and
> make training fail. We exclude runs that fail to produce usable posterior
> samples. Runs with a fluctuating or late-decreasing loss still count if
> they finish and produce usable samples. We provide
> [weights](lv_weights/bayesflow/) for the runs included in the average.

To retrain failed BayesFlow runs with the same seeds in evaluation mode, set
`RETRAIN_MISSING_BAYESFLOW` to `True` in
[the LV runner](../LotkaVoletrra/run_experiment.py).
