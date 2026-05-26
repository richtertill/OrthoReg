# Reproducing the OrthoReg paper

This folder contains the minimum needed to reproduce the main quantitative
results of *OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural
Dynamical Systems*. The specific hyperparameter values
and per-experiment sweep scripts used by the authors are intentionally not
shipped here: the paper's claims follow from the OrthoReg objective itself,
and the released code makes that objective straightforward to invoke.
Reasonable defaults that recover the qualitative behaviour are baked into
[`configs/training/training.yaml`](../configs/training/training.yaml) and the
per-dataset configs.

## What you actually need to run

### Hybrid / SINDy comparison (Tables 1-3, main text)

Three methods on four systems:

```bash
# OrthoReg
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal \
    +training.seed=0

# L2 baseline
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=l2 \
    +training.seed=0

# Pure SINDy baseline
python -m orthoreg.training.train \
    dataset=pendulum model=sindy \
    +training.seed=0
```

Replace `dataset=pendulum` with `dataset=lv`, `dataset=sir`, or
`dataset=duffing` for the other systems, and sweep `+training.seed=0..4` for
five-seed error bars.

### Pure neural baselines (pendulum comparison table)

```bash
python -m orthoreg.training.train_baseline \
    dataset=pendulum model=pinn +training.seed=0

python -m orthoreg.training.train_baseline \
    dataset=pendulum model=universal_ode +training.seed=0
```

### OOD-T3 numbers (optional)

The default `training.retrain_orthogonal_reg_scale` is `1.0`. The paper's
OOD-T3 results on the modified damped pendulum used `0.1`, which relaxes the
orthogonality constraint during stage-2 retraining after the symbolic
structure is fixed. To match that setting:

```bash
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal \
    training.retrain_orthogonal_reg_scale=0.1 \
    +training.seed=0
```

## Resource budget

A single training run on the modified damped pendulum takes about 30-60
minutes on one GPU (NVIDIA A100 / V100 / RTX 3090 class hardware) with the
default 2,000-epoch derivative-fit schedule. Lotka-Volterra and SIR have
similar costs; Duffing is comparable. The full hybrid/SINDy reproduction
matrix is ~60 runs (4 systems x 3 methods x 5 seeds).

## SLURM

[`slurm/template.sh`](slurm/template.sh) is a generic, cluster-agnostic
submission script. Edit the `#SBATCH --partition=...`/`--qos=...`/etc.
placeholders for your site, then submit jobs the same way you would from
the command line:

```bash
sbatch experiments/slurm/template.sh \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal +training.seed=0
```

## Where the metrics live

Each run writes its hyperparameters and metric history to
`project_folder/results/...` under the repository root. The CSV logger
captures train/val/test losses, F1, orthogonality, and OOD derivative MSE
out of the box; readers can inspect these directly with their favourite
analysis tooling. The aggregate-and-make-LaTeX-tables scripts used to
produce the paper's tables are not part of the public release - they are
specific to our internal directory layout.
