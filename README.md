# Summary Learning for Simulation-Based Inference

Code accompanying the manuscript *Neural Estimation of Summary Statistics for
Simulation-based Inference with Non-asymptotic Guarantees*. The repository
contains the four numerical experiments, their configurations, and the
checkpoints and data used by the evaluation scripts.

| Experiment | Directory | Parameters | Observations |
| --- | --- | ---: | --- |
| Gaussian | [`Gaussian/`](Gaussian/) | 5 | Stereographic Gaussian samples |
| SIR | [`SIR/`](SIR/) | 2 | Infection time series |
| Lotka–Volterra | [`LotkaVoletrra/`](LotkaVoletrra/) | 4 | Predator and prey time series |
| Radio | [`Radio/`](Radio/) | 4 | Turin channel measurements |

## Methods

The study evaluates three summary-learning variants, both before and after
applying the local refinement strategy proposed in the manuscript.

- **SMMD** learns a summary network and a conditional posterior generator
  jointly by minimizing sliced maximum mean discrepancy.
- **MMD** uses the same architecture with a multiscale MMD loss.
- **BayesFlow** is the flow-based variant in this comparison. It uses the
  BayesFlow architecture of [Radev et al. (2022)](#references) to learn
  summaries and an amortized posterior. We apply our local refinement
  strategy to its learned summaries as we do for SMMD and MMD.

**Local refinement** starts from the amortized posterior and uses ABC-MCMC
to refine inference for the observed dataset. The Lotka–Volterra **Refine+**
procedure adds a sequential training stage before ABC-MCMC.

The other methods used for comparison are listed below.

- **DNN-ABC** uses a regression estimate of the posterior mean of the
  parameters as the summary for ABC rejection sampling ([Jiang et al., 2017](#references)).
- **NPE / SNPE-C** learn a conditional posterior density from simulated data
  ([Papamakarios & Murray, 2016](#references); [Greenberg et al., 2019](#references)).
- **NLE and TD-NPSE** are amortized methods trained on single
  observations. NLE combines per-observation likelihood estimates, and
  TD-NPSE combines per-observation score estimates. Each supports posterior
  inference for varying numbers of observations without retraining
  ([Papamakarios et al., 2019](#references); [Linhart et al., 2026](#references)).
- **NPE-RS** learns robust summaries for neural posterior estimation under
  model misspecification. We use it for the Radio experiment ([Huang et al., 2023](#references)).

## Reproducing the paper results

The table connects each script to the corresponding part of the manuscript.
The output files contain the metrics and samples used for those comparisons.

| Experiment | Manuscript comparison | Main outputs |
| --- | --- | --- |
| [Gaussian](Gaussian/run_experiment.py) | Table 1, multimodal posterior estimation for 25, 50, and 100 observations | MMD and sliced-Wasserstein distance scaled by √n against a reference posterior. Per-round metrics in `experiment_results.csv` and aggregates in `final_summary.csv`. |
| [Gaussian training budget study](Gaussian/DataSizeDependence/run_experiment.py) | Figure 2, how SMMD and NPE accuracy varies with observation count and training budget | Per-replication and aggregate metrics in `results_raw.csv` and `results_agg.csv`, plus the scaling plot. |
| [SIR](SIR/run_experiment.py) | Table 2, calibration with a small training budget | Coverage, interval lengths, and calibration errors in `calibration_summary.csv`. Coverage curves are based on repeated simulated parameter–observation pairs. |
| [Lotka–Volterra](LotkaVoletrra/run_experiment.py) | Table 3, population ecology under a vague prior | Bias, interval length, and coverage in `final_summary.csv`, with posterior samples and comparisons across the Refine+ stages. |
| [Radio](Radio/run_experiment.py) | Predictive performance with a misspecified simulator | Posterior predictive curves and predictive RMSE. Aggregate metrics are in `rmse_summary_stats.csv`. |


## Repository layout

```text
.
├── Gaussian/
│   └── DataSizeDependence/
├── SIR/
├── LotkaVoletrra/
├── Radio/
├── WEIGHTS/       # Model checkpoints and evaluation data
├── sbi_common/    # Shared losses, network components, and ABC helpers
├── environment/   # Pinned environments
└── tests/         # Unit tests
```

Each task directory contains its own `config.json`, `run_experiment.py`,
simulator in `data_generation.py`, and model definitions in `models/`.
Task-specific metrics and refinement utilities live in `utilities.py`.

## Installation

The environment uses Python 3.13. Choose one of the following installation
options from the repository root.

### Conda

```bash
conda env create -f environment/environment.yml
conda activate sbi-summary
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

The pinned versions are listed in
[`environment/environment.yml`](environment/environment.yml) and
[`environment/requirements.txt`](environment/requirements.txt). See the
[environment notes](environment/README.md) for platform and GPU setup.

The implementation uses **BayesFlow 2.0.8** for the BayesFlow models and
**sbi 0.26.1** for neural density estimation and posterior sampling.
Their software references are listed separately from the method papers
[below](#references).

## Run the experiments

Run the task you want from the repository root.

```bash
# Gaussian defaults to n = 25. For n = 50 or 100, see the config settings below.
python -m Gaussian.run_experiment --config Gaussian/config.json
python -m SIR.run_experiment
python -m LotkaVoletrra.run_experiment
python -m Radio.run_experiment
```

Set `run_mode` in the task's `config.json` to select the workflow.

| Mode | Workflow |
| --- | --- |
| `new_experiment` (default) | Generate data and run the training and evaluation workflow. |
| `evaluation` | Load the supplied checkpoints and data from [`WEIGHTS/`](WEIGHTS/) and run the task's inference, refinement, and evaluation stages as applicable. |

For Gaussian at `n = 50` or `n = 100`, copy `Gaussian/config.json` and set
`n_observation` and `n_observation_list` to `50` and `[50]`, or `100` and
`[100]`, for `new_experiment`. For `evaluation`, set
`evaluation_n_observation_list` to `[50]` or `[100]`. Pass the modified file
with `--config PATH`.

The Gaussian training budget study has its own entry point and configuration.

```bash
python Gaussian/DataSizeDependence/run_experiment.py --config Gaussian/DataSizeDependence/config.json
```

Results are written under each task's `results*/` directories by default.
The main experiments report metrics per round and aggregate them across
completed rounds.

## Configuration

The task configuration sets the simulation budget, methods, and evaluation
options. The exact keys vary by task.

| Key | Meaning |
| --- | --- |
| `models_to_run` | Methods to run |
| `models_config` | Per-method training and inference settings |
| `dataset_size` | Number of simulated training pairs |
| `n_samples_posterior` | Number of posterior samples per estimate |
| `num_rounds`, `n_rounds` | Number of experiment replications in the Gaussian, Lotka–Volterra, and Radio runners |
| `evaluation_num_rounds` | Number of Lotka–Volterra evaluation replications |
| `n_observation_list`, `evaluation_n_observation_list` | Gaussian observation counts for training and checkpoint evaluation |
| `calibration_config` | SIR calibration pairs, grouping, and coverage grid |

## Implementation notes

Configurations specify random seeds and training settings. Floating-point
and sampling differences across devices and library builds can still affect
individual runs. The refinement configuration controls the MCMC proposal
scale and chain length. Acceptance rates and posterior plots can help
diagnose refinement behavior.

## References

**Methods**

- Greenberg, D., Nonnenmacher, M., & Macke, J. (2019). [Automatic posterior transformation for likelihood-free inference](https://proceedings.mlr.press/v97/greenberg19a.html). *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2404–2414.
- Huang, D., Bharti, A., Souza, A., Acerbi, L., & Kaski, S. (2023). [Learning robust statistics for simulation-based inference under model misspecification](https://proceedings.neurips.cc/paper_files/paper/2023/hash/16c5b4102a6b6eb061e502ce6736ad8a-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, 36, 7289–7310.
- Jiang, B., Wu, T.-Y., Zheng, C., & Wong, W. H. (2017). [Learning summary statistic for approximate Bayesian computation via deep neural network](https://www3.stat.sinica.edu.tw/statistica/j27n4/J27N411/J27N411.html). *Statistica Sinica*, 27(4), 1595–1618.
- Linhart, J., Cardoso, G. V., Gramfort, A., Le Corff, S., & Rodrigues, P. L. C. (2026). [Diffusion posterior sampling for simulation-based inference in tall data settings](https://openreview.net/forum?id=cdhfoS6Gyo). *Transactions on Machine Learning Research*.
- Papamakarios, G., & Murray, I. (2016). [Fast ε-free inference of simulation models with Bayesian conditional density estimation](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html). *Advances in Neural Information Processing Systems*, 29, 1028–1036.
- Papamakarios, G., Sterratt, D., & Murray, I. (2019). [Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows](https://proceedings.mlr.press/v89/papamakarios19a.html). *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics*, PMLR 89, 837–848.
- Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2022). [BayesFlow: Learning complex stochastic models with invertible neural networks](https://doi.org/10.1109/TNNLS.2020.3042395). *IEEE Transactions on Neural Networks and Learning Systems*, 33(4), 1452–1466.

**Software**

- Kühmichel, L., Huang, J. M., Pratz, V., Arruda, J., Olischläger, H., Habermann, D., Kucharsky, S., Elsemüller, L., Mishra, A., Bracher, N., Jedhoff, S., Schmitt, M., Bürkner, P.-C., & Radev, S. T. (2026). [BayesFlow 2.0: Multi-backend amortized Bayesian inference in Python](https://arxiv.org/abs/2602.07098). *arXiv:2602.07098*. Package version used here: 2.0.8.
- Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonçalves, P. J., Greenberg, D. S., & Macke, J. H. (2020). [sbi: A toolkit for simulation-based inference](https://joss.theoj.org/papers/10.21105/joss.02505). *Journal of Open Source Software*, 5(52), 2505. Package version used here: 0.26.1.
