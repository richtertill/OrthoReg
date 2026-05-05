"""Hydra entry point for training the hybrid-SINDy model.

Run from the repository root, e.g.

    python -m orthoreg.training.train \
        dataset=pendulum model=hybrid_sindy training.regularization=orthogonal

Hydra resolves the config tree under ``configs/``; every key in those YAMLs is
overridable from the command line (e.g. ``training.lr=5e-3``).
"""
import os

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, open_dict

from orthoreg.data.datamodule import HybridDataModule, setup_dataloaders
from orthoreg.models.exp import HybridExperiment
from orthoreg.paths import RESULT_DIR, TRAINING_DIR
from orthoreg.setup import setup_feature_library, setup_model, setup_trainer


def plot_trajectories(exp, datamodule, method_name, dataset_name,
                      out_dir="results", n_traj=3):
    """Render true vs. predicted trajectories on the ID test set."""
    os.makedirs(out_dir, exist_ok=True)
    target_device = exp.device
    exp.ensure_model_on_device(target_device)

    test_data = datamodule.id_test_data
    y_true = test_data.y[:n_traj].cpu().numpy()
    t = test_data.t.cpu().numpy()

    try:
        with torch.no_grad():
            feature_library = datamodule.feature_library

            def dynamics_fn(t_val, x_val):
                if not isinstance(x_val, torch.Tensor):
                    x_val = torch.tensor(x_val, dtype=torch.float32, device=target_device)
                else:
                    x_val = x_val.to(target_device)
                x_np = x_val.detach().cpu().numpy()
                x_features = torch.tensor(
                    feature_library.transform(x_np),
                    dtype=torch.float32, device=target_device,
                )
                pred_phy, pred_aug = exp.derivative_forward(x_features)
                if exp.cfg.model.hybrid_setting:
                    if exp.cfg.training.regularization == "orthogonal":
                        orth_residual = exp.orthogonal_residual(pred_aug, pred_phy)
                        pred_dy = pred_phy + orth_residual
                    else:
                        pred_dy = pred_phy + pred_aug
                else:
                    pred_dy = pred_phy
                return pred_dy.to(target_device)

            from torchdiffeq import odeint

            y_pred = np.zeros_like(y_true)
            for i in range(n_traj):
                y0 = torch.tensor(y_true[i, 0], dtype=torch.float32, device=exp.device)
                t_tensor = torch.tensor(t, dtype=torch.float32, device=exp.device)
                traj_pred = odeint(
                    dynamics_fn, y0, t_tensor,
                    method="dopri5", rtol=1e-6, atol=1e-8,
                )
                y_pred[i] = traj_pred.detach().cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for state_idx in range(min(2, y_true.shape[2])):
            ax = axes[0, state_idx]
            for traj_idx in range(n_traj):
                ax.plot(t, y_true[traj_idx, :, state_idx],
                        color=f"C{traj_idx}", linestyle="-", linewidth=2,
                        label=f"True {traj_idx + 1}")
                ax.plot(t, y_pred[traj_idx, :, state_idx],
                        color=f"C{traj_idx}", linestyle="--", linewidth=2,
                        label=f"Pred {traj_idx + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"State {state_idx}")
            ax.set_title(f"State {state_idx} evolution")
            ax.grid(True, alpha=0.3)
            if state_idx == 0:
                ax.legend()

        if y_true.shape[2] >= 2:
            ax = axes[1, 0]
            for traj_idx in range(n_traj):
                ax.plot(y_true[traj_idx, :, 0], y_true[traj_idx, :, 1],
                        color=f"C{traj_idx}", linestyle="-", linewidth=2,
                        label=f"True {traj_idx + 1}")
                ax.plot(y_pred[traj_idx, :, 0], y_pred[traj_idx, :, 1],
                        color=f"C{traj_idx}", linestyle="--", linewidth=2,
                        label=f"Pred {traj_idx + 1}")
            ax.set_xlabel(r"$x_0$")
            ax.set_ylabel(r"$x_1$")
            ax.set_title("Phase space")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for traj_idx in range(n_traj):
            state_error = np.linalg.norm(y_true[traj_idx] - y_pred[traj_idx], axis=1)
            ax.plot(t, state_error,
                    color=f"C{traj_idx}", linewidth=2,
                    label=f"Traj {traj_idx + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("State error")
        ax.set_title("Trajectory error")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"{method_name} on {dataset_name}", fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(
            out_dir, f"trajectories_{method_name}_{dataset_name}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Trajectory plot saved to {out_path}")
        final_mse = np.mean((y_true - y_pred) ** 2)
        print(f"Final State MSE: {final_mse:.4f}")

    except Exception as e:
        print(f"Warning: plotting failed but continuing: {e}")

def train(cfg: DictConfig):
    # Seed all RNGs so a given (cfg, seed) reproduces bitwise on the same
    # hardware. ``workers=True`` also seeds Lightning DataLoader workers.
    seed = cfg.training.get("seed", 0)
    seed_everything(seed, workers=True)

    if wandb.run is not None:
        wandb.log({"freeze_epochs": cfg.training.freeze_epochs})
        wandb.log({"freeze_neural": cfg.training.freeze_neural})

    print(f"Start training {cfg.dataset.dataset_name} with hybrid {cfg.model.hybrid_setting}")

    # Setup feature library
    print(f"Setup feature library")
    feature_library = setup_feature_library(cfg)

    # Initialize dataloaders for full training
    print(f"Initialize dataloaders for initial training")
    data_dict = setup_dataloaders(dataset_name=cfg.dataset.dataset_name, cfg=cfg, feature_library=feature_library)
    
    # Get input shape
    input_shape = data_dict['train_data'].y.shape
    
    # Initialize model with correct shape
    net = setup_model(cfg=cfg, train_data_shape=input_shape, feature_library=feature_library, n_sym_features=feature_library.n_output_features_)

    # Initialize experiment
    print(f"Initialize experiment")
    exp = HybridExperiment(net, cfg)
    
    # Initialize datamodule for full training
    print(f"Initialize datamodule for initial training")
    dm = HybridDataModule(cfg, data_dict, feature_library)
    
    print("Setup trainer")
    callbacks, logger, trainer = setup_trainer(cfg)

    print("Phase 1: initial training to discover symbolic structure")
    trainer.fit(exp, dm)
    
    # Test on all test sets
    print(f"Testing initial model on all test sets")
    test_dataloaders = [
        dm.test_dataloader()[i] for i in range(6)  # Get all test dataloaders (including ood_t3)
    ]
    test_results = trainer.test(exp, dataloaders=test_dataloaders)
    
    # Save the symbolic structure (mask)
    initial_mask = exp.net.model_phy.mask_.clone()
    
    # Phase 2: Retrain parameters with fixed structure on t3_extra
    retrain_test_results = None  # Initialize to None
    if cfg.training.retrain:
        print(f"Phase 2: Retraining parameters with fixed structure on t3_extra")
        
        # Create new data dict for retraining
        retrain_data_dict = {
            'train_data': data_dict['ood_test_data_t3_extra'],  # Use t3_extra for training
            'train_data_extra': None,
            'id_test_data': data_dict['id_test_data'],
            'id_test_data_extra': data_dict['id_test_data_extra'],
            'ood_test_data_t2': data_dict['ood_test_data_t2'],
            'ood_test_data_t2_extra': data_dict['ood_test_data_t2_extra'],
            'ood_test_data_t3': data_dict['ood_test_data_t3'],  # Keep t3 for testing
            'ood_test_data_t3_extra': data_dict['ood_test_data_t3_extra'],  # Keep t3_extra for testing
            'W_true': data_dict['W_true']
        }
        
        # Initialize new datamodule for retraining
        print(f"Initializing datamodule for retraining with t3_extra data")
        retrain_dm = HybridDataModule(cfg, retrain_data_dict, feature_library)
        
        with open_dict(cfg):
            cfg.training.lr = cfg.training.retrain_lr
            # Optional: scale the orthogonal regularization weights for the
            # phase-2 fine-tune. The paper uses 0.1 to relax the orthogonality
            # constraint once the symbolic structure is fixed (Section 5,
            # OOD-T3 ablation). Default 1.0 preserves the phase-1 weights.
            if cfg.training.regularization == "orthogonal":
                retrain_reg_scale = cfg.training.get(
                    "retrain_orthogonal_reg_scale", 1.0
                )
                if retrain_reg_scale != 1.0:
                    cfg.training.orthogonal_node_reg_weight *= retrain_reg_scale
                    cfg.training.orthogonal_symbolic_reg_weight *= retrain_reg_scale
                    print(
                        f"Phase 2: scaling orthogonal reg weights by "
                        f"{retrain_reg_scale} "
                        f"(node={cfg.training.orthogonal_node_reg_weight:.6f}, "
                        f"symbolic={cfg.training.orthogonal_symbolic_reg_weight:.6f})"
                    )

        retrain_callbacks = [
            EarlyStopping(
                monitor="train/total_loss",
                patience=cfg.training.retrain_patience,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

        retrain_trainer = L.Trainer(
            max_epochs=cfg.training.n_retrain_epochs,
            callbacks=retrain_callbacks,
            log_every_n_steps=10,
            check_val_every_n_epoch=10,
            logger=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            gradient_clip_val=None,
        )
        
        # Ensure mask is fixed
        exp.net.model_phy.mask_.data = initial_mask
        
        # Retrain with fixed structure on t3_extra
        retrain_trainer.fit(exp, retrain_dm)
        
        # Test on all datasets
        print(f"Testing retrained model on all datasets")
        retrain_test_results = retrain_trainer.test(exp, retrain_dm)
    else:
        print(f"Retraining disabled - skipping Phase 2")
    
    print(f"Training finished")
    # Plot qualitative trajectories
    method_name = (
        "not_hybrid" if not cfg.model.hybrid_setting else
        f"hybrid-{cfg.training.regularization}" if cfg.training.regularization else
        "hybrid"
    )
    
    if torch.cuda.is_available():
        exp.net = exp.net.cuda()

    plot_dir = RESULT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_trajectories(exp, dm, method_name, cfg.dataset.dataset_name, out_dir=str(plot_dir))
    except Exception as e:
        print(f"Warning: Plotting failed but continuing: {e}")
        import traceback
        traceback.print_exc()

    return retrain_test_results, test_results

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Optional W&B logging. Resolved from config first, then $WANDB_ENTITY.
    # If neither is set the trainer uses the CSV logger configured in
    # orthoreg.setup.setup_trainer and never contacts wandb.ai.
    entity = cfg.training.get("wandb_username") or os.environ.get("WANDB_ENTITY")
    if entity is not None:
        wandb.init(
            project=cfg.training.wandb_project,
            entity=entity,
            dir=str(TRAINING_DIR),
            config={
                "lr": cfg.training.lr,
                "hybrid_setting": cfg.model.hybrid_setting,
                "dataset_name": cfg.dataset.dataset_name,
                "regularization": cfg.training.regularization,
                "symbolic_threshold": cfg.training.symbolic_threshold,
                "l2_node_reg_weight": cfg.training.l2_node_reg_weight,
                "l2_symbolic_reg_weight": cfg.training.l2_symbolic_reg_weight,
                "orthogonal_node_reg_weight": cfg.training.orthogonal_node_reg_weight,
                "orthogonal_symbolic_reg_weight": cfg.training.orthogonal_symbolic_reg_weight,
            },
        )

    train(cfg)


if __name__ == "__main__":
    main()