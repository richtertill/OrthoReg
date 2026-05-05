"""Minimal training run that does not require SLURM or W&B.

Edit the constants at the top of the file to switch the dataset, the
model, the regularisation, or the training budget. Then::

    python examples/quick_train.py

The script reuses the same Hydra configs as ``orthoreg.training.train``;
this is just a thin Python wrapper so you do not have to remember the
override syntax.
"""

import os
import sys

# Hydra resolves config paths relative to the repository root, so make
# sure we run from there even when the user invoked this from elsewhere.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import lightning as L  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from orthoreg.data.datamodule import HybridDataModule, setup_dataloaders  # noqa: E402
from orthoreg.models.exp import HybridExperiment  # noqa: E402
from orthoreg.setup import setup_feature_library, setup_model, setup_trainer  # noqa: E402


# --- Edit these to customise the run ----------------------------------------
DATASET = "pendulum"            # one of: pendulum, lv, sir, duffing
MODEL = "hybrid_sindy"          # one of: hybrid_sindy, sindy
REGULARIZATION = "orthogonal"   # one of: orthogonal, l2, none
NUM_EPOCHS = 100                # 100 for a quick smoke test, 2000 for paper
ORTHO_REG_WEIGHT = 5.0          # OrthoReg penalty strength
# ----------------------------------------------------------------------------

print("OrthoReg quick-train")
print(f"  dataset:         {DATASET}")
print(f"  model:           {MODEL}")
print(f"  regularization:  {REGULARIZATION}")
print(f"  epochs:          {NUM_EPOCHS}")
print(f"  ortho lambda:    {ORTHO_REG_WEIGHT}")

GlobalHydra.instance().clear()
configs_path = os.path.join(repo_root, "configs")
initialize_config_dir(config_dir=configs_path, version_base=None)
cfg = compose(
    config_name="config",
    overrides=[
        f"dataset={DATASET}",
        f"model={MODEL}",
        f"training.regularization={REGULARIZATION}",
        f"training.n_derivative_epochs={NUM_EPOCHS}",
        f"training.orthogonal_node_reg_weight={ORTHO_REG_WEIGHT}",
    ],
)
OmegaConf.set_struct(cfg, False)

print("[1/5] feature library")
feature_library = setup_feature_library(cfg)

print("[2/5] data")
data_dict = setup_dataloaders(
    dataset_name=cfg.dataset.dataset_name,
    cfg=cfg,
    feature_library=feature_library,
)
dm = HybridDataModule(cfg, data_dict, feature_library)
n_features = feature_library.n_output_features_

print("[3/5] model")
net = setup_model(cfg, data_dict["train_data"].y.shape, feature_library, n_features)
exp = HybridExperiment(net, cfg)

print("[4/5] trainer")
callbacks, logger, trainer = setup_trainer(cfg)
print(f"      cuda available: {torch.cuda.is_available()}")

print("[5/5] training")
trainer.fit(exp, datamodule=dm)
print("Training complete.")

print("Testing")
test_results = trainer.test(exp, datamodule=dm)
for key, value in test_results[0].items():
    print(f"  {key}: {value:.6f}")

print("Done. Artifacts written under project_folder/.")
