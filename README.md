# OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems

[![CI](https://github.com/richtertill/OrthoReg/actions/workflows/ci.yml/badge.svg)](https://github.com/richtertill/OrthoReg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OrthoReg directly penalizes overlap between the symbolic and neural components
of a hybrid dynamical-systems model, so symbolic structure is not absorbed by
the neural residual. The result is a complementary decomposition: the
symbolic part captures what the library can express, and the neural part
captures what remains.

This repository contains the code for *OrthoReg: Orthogonal Regularization
for Hybrid Symbolic-Neural Dynamical Systems*. It ships the training
pipeline, the five benchmark systems used in the paper, the baselines, and
the unit tests.

## What is in here

```
orthoreg/                # Python package
  data/datasets/         # the 5 paper systems (pendulum, duffing, lotka_volterra, sir, helmholtz)
  models/                # hybrid-SINDy + neural augmentation; PINN, Universal ODE, STLSQ baselines
  regularization/        # OrthoReg penalty and projection utilities
  training/              # Hydra/Lightning training entry point
configs/                 # Hydra configs for dataset / model / training
examples/                # Self-contained examples (no SLURM, no wandb)
experiments/             # Commands for the paper's experiments
tests/                   # Unit, known-answer and end-to-end smoke tests
```

## Method in one equation

OrthoReg adds an explicit empirical-orthogonality penalty to a sparse
hybrid-SINDy objective. Given a symbolic library $`\{\phi_j\}_{j=1}^M`$ and
a neural augmentation $`\hat{f}_{\mathrm{aug}}(\,\cdot\,;\vartheta)`$
evaluated on the observed states $`\mathcal{D}=\{x_i\}_{i=1}^N`$, the
penalty is

$$\mathcal{L}_{\mathrm{reg}}^{\perp}(\vartheta) = \lambda \sum_{j=1}^M \langle \hat{f}_{\mathrm{aug}}, \phi_j \rangle_{\mathcal{D}}^2,$$

where $`\langle f, g \rangle_{\mathcal{D}} = \tfrac{1}{N} \sum_i f(x_i)^\top g(x_i)`$
is the empirical inner product and each library slot $`\phi_j`$ is one
feature placed in one state dimension. The model predicts
$`\hat f(x) = \sum_j w_j \phi_j(x) + \hat f_{\mathrm{aug}}(x)`$ and every step
minimizes the joint objective (Algorithm 1 of the paper)

$$\frac{1}{B}\sum_{i} \lVert y_i - \hat f(x_i) \rVert^2 + \mu \lVert w \rVert_1 + \mathcal{L}_{\mathrm{reg}}^{\perp}(\vartheta).$$

A vanishing penalty implies
$`\hat{f}_{\mathrm{aug}} \perp \mathrm{span}\{\phi_1, \ldots, \phi_M\}`$. The
$L^2$ baseline replaces the penalty with
$`\lambda \lVert \hat f_{\mathrm{aug}} \rVert_{\mathcal{D}}^2`$; the penalty is
implemented in [`orthoreg/regularization/orthoreg.py`](orthoreg/regularization/orthoreg.py).

## Paper

Till Richter and Niki Kilbertus. *OrthoReg: Orthogonal Regularization for
Hybrid Symbolic-Neural Dynamical Systems.* NeurIPS 2026.
[arXiv:2606.19145](https://arxiv.org/abs/2606.19145)

## Install

```bash
git clone https://github.com/richtertill/OrthoReg.git
cd OrthoReg
pip install -e .
```

Python 3.10 or newer is required. PyTorch / CUDA is auto-detected at runtime;
training will use a GPU if one is available.

## Quickstart

A single training run on the paper's modified damped pendulum:

```bash
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal
```

Swap `dataset=pendulum` for any of `duffing`, `lotka_volterra`, `sir`,
`helmholtz`. Swap
`training.regularization` for `l2` (L^2-regularized hybrid baseline) or
`none` (pure SINDy, paired with `model=sindy`). All knobs are
[Hydra](https://hydra.cc/) overrides; the full default configuration lives in
[configs/](configs/).

The canonical SINDy baseline (PySINDy STLSQ, threshold 0.045, same library
and splits) and the pure-neural baselines run through a second entry point:

```bash
python -m orthoreg.training.train_baseline dataset=pendulum model=stlsq
python -m orthoreg.training.train_baseline dataset=pendulum model=pinn
python -m orthoreg.training.train_baseline dataset=pendulum model=universal_ode
```

PINN-style and UODE-style baselines with OrthoReg's library, next to OrthoReg:

```bash
# OrthoReg
python -m orthoreg.training.train dataset=pendulum model=hybrid_sindy training.regularization=orthogonal training.orthogonal_node_reg_weight=0.02 training.orthogonal_symbolic_reg_weight=0.012 model.hidden_dim=32 model.num_layers=3
# PINN-style
python -m orthoreg.training.train_baseline dataset=pendulum model=pinn model.physics=library model.hidden_dim=32 model.num_layers=3 training.orthogonal_symbolic_reg_weight=0.012
# UODE-style
python -m orthoreg.training.train_baseline dataset=pendulum model=universal_ode model.physics=library model.hidden_dim=32 model.num_layers=3 training.orthogonal_symbolic_reg_weight=0.012
```

`model.physics=library` uses OrthoReg's library with learnable, L1-penalized
coefficients; the default `model.physics=known` uses the base pendulum
equation.

## Systems

All five systems share the library degree-2 polynomials (no bias) plus
Fourier features at unit frequency. `n_out_of_library_terms` (K) in each
dataset config counts the true terms the library cannot express; it enters
the full-support recall and F1.

| `dataset=`       | System                                       | Out-of-library terms (K)                       |
| ---------------- | -------------------------------------------- | ---------------------------------------------- |
| `pendulum`       | Modified damped pendulum                     | cos(3 x0), exp(-x0^2), tanh(x1) (3)            |
| `duffing`        | Unforced Duffing oscillator (bistable)       | c x0^3 (1)                                     |
| `lotka_volterra` | Lotka-Volterra with modulated coupling       | one modulated coupling per equation (2)        |
| `sir`            | SIR with memory-modulated terms              | one modulated term per compartment (3)         |
| `helmholtz`      | Damped Helmholtz oscillator with corrugation | A cos(Omega x0) (1)                            |

Test metrics are reported on the ID, OOD-T2 (initial-condition shift) and
OOD-T3 (parameter shift) splits and their time-extrapolated versions:
derivative and state MSE, and support precision, recall, F1, false-positive
rate and full-support recall / F1.

To run without configuring W&B, the [`examples/`](examples/) folder
has two scripts:

```bash
# Demonstrates the orthogonal projection on a synthetic damped pendulum.
python examples/simple_pendulum_example.py

# Minimal copy-paste training script (edit the constants at the top).
python examples/quick_train.py
```

## Running the paper's experiments

See [experiments/README.md](experiments/README.md) for the commands.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The tests cover the training-time penalties against hand-computed values,
the training-step objective (data fit + L1 + penalty) and the hybrid
prediction, the support-recovery metrics on a hand-built example, data
generation and the ground-truth support of every system, the offline
projection utilities, and short CPU training runs of every system with
the pure, L2, OrthoReg and STLSQ models.

## Logging

Training falls back to a local CSV logger by default. To log to Weights &
Biases instead, set `WANDB_ENTITY` in the environment (or
`training.wandb_username` in the config) and the trainer will pick it up.

Generated datasets, checkpoints, logs and plots go to `project_folder/` in
the repository root; set `ORTHOREG_OUTPUT_DIR` to write them elsewhere.

## Citation

```bibtex
@inproceedings{richter2026orthoreg,
    title     = {OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems},
    author    = {Richter, Till and Kilbertus, Niki},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2026},
    eprint    = {2606.19145},
    archivePrefix = {arXiv},
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgements

OrthoReg builds on:

- **APHYNITY** ([Yin et al., ICLR 2021](https://github.com/yuan-yin/APHYNITY)) for the hybrid additive decomposition.
- **PySINDy** ([de Silva et al., 2020](https://github.com/dynamicslab/pysindy)) for the symbolic library and sparse-regression utilities.

## License

MIT - see [LICENSE](LICENSE).
