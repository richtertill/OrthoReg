"""Shared test setup.

Generated datasets, checkpoints and logs go to a throw-away directory so
the test suite never writes into ``project_folder/`` of the checkout. The
environment variable must be set before :mod:`orthoreg.paths` is imported.
"""

import os
import tempfile

os.environ.setdefault("ORTHOREG_OUTPUT_DIR", tempfile.mkdtemp(prefix="orthoreg_tests_"))

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@pytest.fixture
def compose_cfg():
    """Compose the Hydra config tree from ``configs/`` with overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    def _compose(overrides):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=list(overrides))
        OmegaConf.set_struct(cfg, False)
        return cfg

    yield _compose
    GlobalHydra.instance().clear()
