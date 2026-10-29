import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# Generated datasets, checkpoints and results go under ``project_folder/`` in
# the repository root unless ``ORTHOREG_OUTPUT_DIR`` points elsewhere.
OUTPUT_DIR = Path(os.environ.get("ORTHOREG_OUTPUT_DIR", ROOT / "project_folder"))

DATA_DIR = OUTPUT_DIR / "datasets"
TRAINING_DIR = OUTPUT_DIR / "experiments"
RESULT_DIR = OUTPUT_DIR / "results"
