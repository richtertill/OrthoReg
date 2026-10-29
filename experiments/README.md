# Running the experiments of the OrthoReg paper

Commands for the methods and systems of *OrthoReg: Orthogonal Regularization
for Hybrid Symbolic-Neural Dynamical Systems*. Every setting is a Hydra
override.

## What you actually need to run

### Hybrid / SINDy comparison

Three methods on five systems:

```bash
# OrthoReg
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal \
    training.seed=0

# L2 baseline
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=l2 \
    training.seed=0

# Pure SINDy baseline
python -m orthoreg.training.train \
    dataset=pendulum model=sindy \
    training.seed=0
```

Replace `dataset=pendulum` with `dataset=duffing`, `dataset=lotka_volterra`,
`dataset=sir` or `dataset=helmholtz` for the other systems, and sweep
`training.seed=0..4` for five-seed error bars.

### Canonical SINDy (STLSQ) baseline

```bash
python -m orthoreg.training.train_baseline \
    dataset=pendulum model=stlsq training.seed=0
```

PySINDy `STLSQ(threshold=0.045, normalize_columns=False)` on second-order
finite differences, same library and splits, scored with the same metrics.
With `training.retrain=true` (default) the selected support is refit by
least squares on the OOD-T3 extrapolation trajectories and scored again.

### PINN / Universal ODE baselines

```bash
# OrthoReg
python -m orthoreg.training.train \
    dataset=pendulum model=hybrid_sindy \
    training.regularization=orthogonal \
    training.orthogonal_node_reg_weight=0.02 \
    training.orthogonal_symbolic_reg_weight=0.012 \
    model.hidden_dim=32 model.num_layers=3 training.seed=0

# PINN-style
python -m orthoreg.training.train_baseline \
    dataset=pendulum model=pinn model.physics=library \
    model.hidden_dim=32 model.num_layers=3 \
    training.orthogonal_symbolic_reg_weight=0.012 training.seed=0

# UODE-style
python -m orthoreg.training.train_baseline \
    dataset=pendulum model=universal_ode model.physics=library \
    model.hidden_dim=32 model.num_layers=3 \
    training.orthogonal_symbolic_reg_weight=0.012 training.seed=0
```

`model.physics=library` uses OrthoReg's library `Theta(x) xi` with learnable,
L1-penalized coefficients; the default `model.physics=known` uses the base
pendulum equation `-omega0^2 sin(x0) - alpha x1`.

## Resource budget

A single training run on the modified damped pendulum takes about 30-60
minutes on one GPU (NVIDIA A100 / V100 / RTX 3090 class hardware) with the
default 2,000-epoch derivative-fit schedule. The other systems have similar
costs. The full hybrid/SINDy run matrix is 75 runs (5 systems x 3
methods x 5 seeds); STLSQ takes seconds per run.

## Where the metrics live

Each run writes its hyperparameters and metric history under
`project_folder/` in the repository root (or `$ORTHOREG_OUTPUT_DIR`). The
CSV logger captures train losses and, per test split, derivative and state
MSE, support precision / recall / F1 / false-positive rate, full-support
recall / F1 and the exploratory component cosine.
