# OrthoReg examples

Two self-contained scripts. Neither requires SLURM or a W&B account.

| Script | What it demonstrates |
| --- | --- |
| `simple_pendulum_example.py` | Numerical verification of the OrthoReg geometry on synthetic pendulum data. Builds a polynomial library, constructs a "neural" candidate that deliberately overlaps the library, and shows that `orthogonalize_function` returns a residual whose empirical inner product with every library term is `~0`. This is the offline counterpart of the training-time penalty implemented in `orthoreg/regularization/orthoreg.py`. |
| `quick_train.py` | Minimal end-to-end training run that bypasses the Hydra command-line surface. Edit the constants at the top (dataset, model, regularization, epochs, OrthoReg lambda) and run `python examples/quick_train.py` from the repo root. Reuses the same configs as `python -m orthoreg.training.train`. |

Run them from the repository root:

```bash
python examples/simple_pendulum_example.py
python examples/quick_train.py
```
