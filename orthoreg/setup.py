"""Setup utilities for OrthoReg experiments."""
import os
from typing import Tuple
from omegaconf import DictConfig
from orthoreg.paths import TRAINING_DIR
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import lightning as L
import torch
import pysindy as ps
import torch.nn as nn
from orthoreg.models.networks import (
    PhysicalModel,
    MLP
)
from orthoreg.utils import init_weights
from orthoreg.models.forecasters import Forecaster


def get_experiment_and_checkpoint_dir(cfg: DictConfig) -> Tuple[str, str]:
    """Create experiment and checkpoint directories."""
    folder_name = f"{cfg.dataset.dataset_name}_{cfg.model.hybrid_setting}_{cfg.model.symbolic_regression}"
    
    experiment_dir = os.path.join(TRAINING_DIR, folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir


def setup_trainer(cfg: DictConfig):
    """Setup training callbacks and logger."""
    experiment_dir, checkpoint_dir = get_experiment_and_checkpoint_dir(cfg)
    
    callbacks = [
        # Single checkpoint for best model
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            monitor='train/total_loss',
            mode='min',
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Only create wandb logger if wandb is available and configured
    try:
        import wandb
        if hasattr(cfg.training, 'logger') and hasattr(cfg.training.logger, 'project'):
            logger = WandbLogger(
                project=cfg.training.logger.project,
                name=f"{cfg.dataset.dataset_name}_{cfg.model.hybrid_setting}_{cfg.model.symbolic_regression}",
                config=dict(cfg)
            )
        else:
            # Fallback to CSV logger if wandb config is missing
            logger = CSVLogger(save_dir=experiment_dir, name="logs")
    except (ImportError, AttributeError):
        # Fallback to CSV logger if wandb is not available
        logger = CSVLogger(save_dir=experiment_dir, name="logs")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = L.Trainer(
        max_epochs=cfg.training.n_derivative_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=None,
        num_sanity_val_steps=0,
    )

    return callbacks, logger, trainer


def setup_feature_library(cfg: DictConfig):
    """Setup the feature library for symbolic regression."""
    polynomial_library = ps.PolynomialLibrary(degree=cfg.training.polynomial_degree, include_bias=False)
    fourier_library = ps.FourierLibrary(n_frequencies=cfg.training.n_frequencies)
    
    # Combine libraries
    feature_library = ps.ConcatLibrary([polynomial_library, fourier_library])
    return feature_library


def setup_model(cfg: DictConfig, train_data_shape: Tuple[int, ...], feature_library, n_sym_features: int) -> nn.Module:
    """Initialize the hybrid forecasting model."""
    # Determine device
    if hasattr(cfg.training, 'device') and cfg.training.device != 'auto':
        if 'cuda' in cfg.training.device and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Setting up model on device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Number of state variables -> derived from the data so the dataset
    # config never has to declare it (it is a system property, not a model
    # hyperparameter).
    n_states = train_data_shape[-1]

    # Initialize physical model
    model_phy = PhysicalModel(
        feature_library=feature_library,
        device=device,
        cfg=cfg,
        n_states=n_states,
    )

    # Augmentation MLP shape is read from the model config. The OLD repo
    # mixed ``cfg.dataset.forecaster.hidden_dim`` (used by hybrid_sindy /
    # sindy) with ``cfg.model.hidden_dim`` (used by PINN / UODE); the new
    # surface unifies both on ``cfg.model.hidden_dim``.
    hidden_dim = cfg.model.get('hidden_dim', 64)
    num_layers = cfg.model.get('num_layers', 3)

    print(f"Setting up models with dimensions:")
    print(f"  Input features: {n_sym_features}")
    print(f"  Output states:  {n_states}")
    print(f"  Hidden dim:     {hidden_dim}")
    print(f"  Num layers:     {num_layers}")

    model_aug = MLP(
        state_c=n_sym_features,
        hidden=hidden_dim,
        num_layers=num_layers,
        out_c=n_states,
    )

    init_weights(model_aug, init_type='orthogonal', init_gain=0.01)

    net = Forecaster(
        model_phy=model_phy,
        model_aug=model_aug,
        hybrid_setting=cfg.model.hybrid_setting,
        method=cfg.training.get('ode_method', 'rk4'),
        options=cfg.training.get('ode_options', {'step_size': 0.01})
    )
    
    # Move to device
    net = net.to(device)

    print('MLP in Neural ODE:')
    print(model_aug)
    print(f'Model device: {next(net.parameters()).device}')

    return net

