from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT / "project_folder" / "datasets"
TRAINING_DIR = ROOT / "project_folder" / "experiments"
RESULT_DIR = ROOT / "project_folder" / "results"