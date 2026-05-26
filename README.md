# OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems

[![CI](https://github.com/richtertill/OrthoReg/actions/workflows/ci.yml/badge.svg)](https://github.com/richtertill/OrthoReg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OrthoReg directly penalises overlap between the symbolic and neural components
of a hybrid dynamical-systems model, so symbolic structure is not absorbed by
the neural residual. The result is a complementary decomposition: the
symbolic part captures what the library can express, and the neural part
captures what remains.

This repository contains the code for *OrthoReg: Orthogonal Regularization
for Hybrid Symbolic-Neural Dynamical Systems*. It ships the training
pipeline, the four benchmark systems used in the paper, and the unit tests
covering the orthogonal projection itself.

## What is in here

```
orthoreg/                # Python package
  data/datasets/         # the 4 paper systems (pendulum, lv, sir, duffing)
  models/                # hybrid-SINDy + neural augmentation, PINN, Universal ODE
  regularization/        # OrthoReg penalty and projection utilities
  training/              # Hydra/Lightning training entry point
configs/                 # Hydra configs for dataset / model / training / launcher
examples/                # Self-contained examples (no SLURM, no wandb)
experiments/             # Reproduction guide + a generic SLURM template
tests/                   # Unit tests for the orthogonal projection (the core contribution)
```

## Method in one equation

OrthoReg adds an explicit empirical-orthogonality penalty to a sparse
hybrid-SINDy objective. Given a symbolic library $`\{\phi_j\}_{j=1}^M`$ and
a neural augmentation $`\hat{f}_{\mathrm{aug}}(\,\cdot\,;\vartheta)`$
evaluated on the observed states $`\mathcal{D}=\{x_i\}_{i=1}^N`$, the
penalty is

$$\mathcal{L}_{\mathrm{reg}}^{\perp}(\vartheta) = \lambda \sum_{j=1}^M \langle \hat{f}_{\mathrm{aug}}, \phi_j \rangle_{\mathcal{D}}^2,$$

where $`\langle f, g \rangle_{\mathcal{D}} = \tfrac{1}{N} \sum_i f(x_i)^\top g(x_i)`$
is the empirical inner product. The penalty is added to the standard
fit + L1 sparsity loss; vanishing penalty implies
$`\hat{f}_{\mathrm{aug}} \perp \mathrm{span}\{\phi_1, \ldots, \phi_M\}`$.

The paper makes a scoped empirical claim: OrthoReg helps most when the
symbolic library is partially misspecified, in which regime it improves
symbolic recovery and reduces out-of-distribution derivative error compared
to $L^2$-regularised hybrid models, at the cost of some in-distribution fit.
See the paper for the exact statements and the systems on which the trade-off
is most pronounced.

## Paper

Preprint link will be added upon arXiv release.

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

Swap `dataset=pendulum` for any of `lv`, `sir`, `duffing`. Swap
`training.regularization` for `l2` (L^2-regularised hybrid baseline) or
`none` (pure SINDy, paired with `model=sindy`). All knobs are
[Hydra](https://hydra.cc/) overrides; the full default configuration lives in
[configs/](configs/).

To run without configuring SLURM / W&B, the [`examples/`](examples/) folder
has two scripts:

```bash
# Demonstrates the orthogonal projection on a synthetic damped pendulum.
python examples/simple_pendulum_example.py

# Minimal copy-paste training script (edit the constants at the top).
python examples/quick_train.py
```

## Reproducing the paper

`experiments/` ships:

- a one-page reproduction guide ([experiments/README.md](experiments/README.md)),
- a generic SLURM template ([experiments/slurm/template.sh](experiments/slurm/template.sh)).

We do not ship per-experiment hyperparameter sweep scripts or the
aggregate-and-make-LaTeX-tables tooling - those are tied to our internal
directory layout. The paper's claims follow from the OrthoReg objective
itself, and [`configs/training/training.yaml`](configs/training/training.yaml)
contains reasonable defaults that recover the qualitative behaviour. The
optimiser, schedule, and regularisation strengths are exposed as Hydra
overrides so the values used in any specific table can be reproduced from
the command line.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The unit tests cover the orthogonal projection mathematics (the core
contribution), the training-time `orthoreg_penalty`, hybrid-model forward
passes, the SINDy feature library wiring, and a lightweight pendulum
dataset smoke test.

## Logging

Training falls back to a local CSV logger by default. To log to Weights &
Biases instead, set `WANDB_ENTITY` in the environment (or
`training.wandb_username` in the config) and the trainer will pick it up.

## Citation

```bibtex
@misc{anonymous2026orthoreg,
    title  = {OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems},
    author = {Anonymous Authors},
    year   = {2026},
}
```

## Acknowledgements

OrthoReg builds on:

- **APHYNITY** ([Yin et al., ICLR 2021](https://github.com/yuan-yin/APHYNITY)) for the hybrid additive decomposition.
- **PySINDy** ([de Silva et al., 2020](https://github.com/dynamicslab/pysindy)) for the symbolic library and sparse-regression utilities.

## License

MIT - see [LICENSE](LICENSE).
