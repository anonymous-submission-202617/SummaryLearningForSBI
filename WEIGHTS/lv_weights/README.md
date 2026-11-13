# Lotka–Volterra BayesFlow checkpoints

In `bayesflow/`, each available seed has a summary network (`*.weights.h5`) and an inference
network (`*_inference.pt`). Missing seeds correspond to runs without usable
checkpoints after unstable training. The paper discusses how the KL loss can
cause exploding gradients and prevent BayesFlow from training successfully.
Evaluation skips missing seeds by default and averages over successful runs.

To attempt those seeds locally, set `RETRAIN_MISSING_BAYESFLOW = True` in
[`LotkaVoletrra/run_experiment.py`](../../LotkaVoletrra/run_experiment.py)
and run in evaluation mode. The runner still loads existing checkpoint pairs
and trains only the missing replications from scratch using their seed
labels. Retraining can encounter the same instability.
