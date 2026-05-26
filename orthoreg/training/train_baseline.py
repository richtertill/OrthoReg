"""Hydra entry point for the PINN and Universal ODE baselines.

Run from the repository root, e.g.

    python -m orthoreg.training.train_baseline dataset=pendulum model=pinn
"""

import os

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from orthoreg.data.datamodule import HybridDataModule, setup_dataloaders
from orthoreg.models.baselines.pinn import PINNExperiment
from orthoreg.models.baselines.universal_ode import UniversalODEExperiment
from orthoreg.paths import RESULT_DIR, TRAINING_DIR
from orthoreg.setup import setup_feature_library, setup_trainer
from orthoreg.utils import get_wandb

def plot_trajectories(exp, datamodule, method_name, dataset_name,
                      out_dir="results", n_traj=3):
    """Render true vs. predicted trajectories for baseline models."""
    os.makedirs(out_dir, exist_ok=True)
    target_device = exp.device

    test_data = datamodule.id_test_data
    y_true = test_data.y[:n_traj].cpu().numpy()
    t = test_data.t.cpu().numpy()

    try:
        with torch.no_grad():
            y_pred = np.zeros_like(y_true)
            for i in range(n_traj):
                y0 = torch.tensor(y_true[i, 0], dtype=torch.float32, device=target_device)
                t_tensor = torch.tensor(t, dtype=torch.float32, device=target_device)

                if hasattr(exp, "pinn_net"):
                    def dynamics_fn(t_val, x_val):
                        if not isinstance(t_val, torch.Tensor):
                            t_val = torch.tensor(t_val, dtype=torch.float32, device=target_device)
                        if not isinstance(x_val, torch.Tensor):
                            x_val = torch.tensor(x_val, dtype=torch.float32, device=target_device)
                        if t_val.dim() == 0:
                            t_val = t_val.unsqueeze(0)
                        network_input = torch.cat(
                            [t_val] + [x_val[k:k + 1] for k in range(x_val.shape[0])],
                            dim=0,
                        ).unsqueeze(0)
                        return exp.pinn_net(network_input).squeeze(0)
                elif hasattr(exp, "uode_net"):
                    def dynamics_fn(t_val, x_val):
                        if not isinstance(t_val, torch.Tensor):
                            t_val = torch.tensor(t_val, dtype=torch.float32, device=target_device)
                        if not isinstance(x_val, torch.Tensor):
                            x_val = torch.tensor(x_val, dtype=torch.float32, device=target_device)
                        return exp.full_dynamics(t_val, x_val)
                else:
                    continue

                from torchdiffeq import odeint
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

def train_baseline(cfg: DictConfig):
    """Train baseline model (PINN or Universal ODE)."""
    wandb = get_wandb()
    if wandb is not None and wandb.run is not None:
        wandb.log({"freeze_epochs": cfg.training.freeze_epochs})
        wandb.log({"freeze_neural": cfg.training.freeze_neural})

    print(f"Start training {cfg.dataset.dataset_name} with baseline {cfg.model.baseline_type}")

    # Setup feature library (not used by baselines but needed for datamodule)
    print(f"Setup feature library")
    feature_library = setup_feature_library(cfg)

    # Initialize dataloaders for full training
    print(f"Initialize dataloaders for initial training")
    data_dict = setup_dataloaders(dataset_name=cfg.dataset.dataset_name, cfg=cfg, feature_library=feature_library)
    
    # Initialize baseline experiment
    print(f"Initialize baseline experiment")
    if cfg.model.baseline_type == 'pinn':
        exp = PINNExperiment(cfg, feature_library)
    elif cfg.model.baseline_type == 'universal_ode':
        exp = UniversalODEExperiment(cfg, feature_library)
    else:
        raise ValueError(f"Unknown baseline type: {cfg.model.baseline_type}")
    
    # Initialize datamodule for full training
    print(f"Initialize datamodule for initial training")
    dm = HybridDataModule(cfg, data_dict, feature_library)
    
    print("Setup trainer")
    callbacks, logger, trainer = setup_trainer(cfg)

    print("Training baseline model")
    trainer.fit(exp, dm)
    
    # Test on all test sets
    print(f"Testing baseline model on all test sets")
    test_dataloaders = [
        dm.test_dataloader()[i] for i in range(6)  # Get all test dataloaders (including ood_t3)
    ]
    test_results = trainer.test(exp, dataloaders=test_dataloaders)
    
    print(f"Training finished")
    
    # Plot qualitative trajectories
    method_name = cfg.model.baseline_type
    
    if torch.cuda.is_available():
        exp = exp.cuda()

    plot_dir = RESULT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_trajectories(exp, dm, method_name, cfg.dataset.dataset_name, out_dir=str(plot_dir))
    except Exception as e:
        print(f"Warning: Plotting failed but continuing: {e}")
        import traceback
        traceback.print_exc()

    return test_results, test_results


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    entity = cfg.training.get("wandb_username") or os.environ.get("WANDB_ENTITY")
    if entity is not None:
        wandb = get_wandb()
        if wandb is None:
            print(
                "Warning: WANDB_ENTITY is set but wandb is not installed. "
                "Install with pip install -e '.[logging]' or unset WANDB_ENTITY."
            )
        else:
            wandb.init(
                project=cfg.training.wandb_project,
                entity=entity,
                dir=str(TRAINING_DIR),
                config={
                    "lr": cfg.training.lr,
                    "baseline_type": cfg.model.baseline_type,
                    "dataset_name": cfg.dataset.dataset_name,
                },
            )
    train_baseline(cfg)


if __name__ == "__main__":
    main()

