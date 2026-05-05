"""
Physics-Informed Neural Networks (PINNs) implementation for pendulum dynamics.

Based on Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework 
for solving forward and inverse problems involving nonlinear partial differential equations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import lightning as L


class PINNNetwork(nn.Module):
    """Physics-Informed Neural Network for pendulum dynamics.

    Learns the map from (t, theta, theta_dot) to (theta_dot, theta_ddot)
    subject to the damped-pendulum physics constraint.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 4,
                 output_dim: int = 2, activation: str = 'tanh'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Linear(hidden_dim, hidden_dim)
            ])
        
        # Output layer
        layers.extend([
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation, nn.Tanh())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (batch, 3) = (t, theta, theta_dot).

        Returns:
            tensor of shape (batch, 2) = (theta_dot, theta_ddot).
        """
        return self.network(x)


class PINNExperiment(L.LightningModule):
    """
    Lightning module for training PINN on pendulum dynamics.
    """
    
    def __init__(self, cfg, feature_library=None):
        super().__init__()
        self.cfg = cfg
        self.feature_library = feature_library
        
        input_dim = 3  # (t, theta, theta_dot)
        output_dim = 2  # (theta_dot, theta_ddot)
        hidden_dim = cfg.model.get('hidden_dim', 64)
        num_layers = cfg.model.get('num_layers', 4)
        activation = cfg.model.get('activation', 'tanh')
        
        # Initialize PINN network
        self.pinn_net = PINNNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            activation=activation
        )
        
        # Physics parameters (learnable or fixed)
        self.omega0 = nn.Parameter(torch.tensor(1.0))  # Natural frequency
        self.alpha = nn.Parameter(torch.tensor(0.2))   # Damping coefficient
        
        # Loss weights
        self.data_weight = cfg.training.get('data_weight', 1.0)
        self.physics_weight = cfg.training.get('physics_weight', 1.0)
        self.bc_weight = cfg.training.get('bc_weight', 1.0)
        
        # Training parameters
        self.lr = cfg.training.lr
        self.automatic_optimization = False
        
        # Store training data for physics loss computation
        self.training_data = None
        
    def set_training_data(self, t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor):
        """Set training data for physics loss computation."""
        self.training_data = {
            't': t,
            'y': y,
            'dy': dy
        }
    
    def physics_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Physics loss for the damped-pendulum equation.

        Enforces the pendulum ODE
            theta_ddot = -omega_0^2 * sin(theta) - alpha * theta_dot

        Args:
            t: time tensor of shape (batch, 1).
            y: state tensor of shape (batch, 2) = (theta, theta_dot).
        """
        theta = y[:, 0:1]
        theta_dot = y[:, 1:2]

        network_input = torch.cat([t, theta, theta_dot], dim=1)
        pred = self.pinn_net(network_input)
        pred_theta_dot = pred[:, 0:1]
        pred_theta_ddot = pred[:, 1:2]

        eq1 = pred_theta_dot - theta_dot
        expected_theta_ddot = (
            -self.omega0 ** 2 * torch.sin(theta) - self.alpha * theta_dot
        )
        eq2 = pred_theta_ddot - expected_theta_ddot

        return torch.mean(eq1 ** 2) + torch.mean(eq2 ** 2)
    
    def data_loss(self, t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            t: Time tensor
            y: State tensor
            dy: Derivative tensor (ground truth)
            
        Returns:
            Data loss tensor
        """
        # Prepare input for network
        theta = y[:, 0:1]
        theta_dot = y[:, 1:2]
        network_input = torch.cat([t, theta, theta_dot], dim=1)
        
        # Get network predictions
        pred = self.pinn_net(network_input)
        
        # Compute MSE loss
        data_loss = F.mse_loss(pred, dy)
        
        return data_loss
    
    def boundary_condition_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss (if any).
        For pendulum, we might want to enforce periodicity or other constraints.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            Boundary condition loss tensor
        """
        # For now, return zero loss (no specific boundary conditions)
        # This can be extended for specific boundary conditions
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def total_loss(self, t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss combining data, physics, and boundary condition losses.
        
        Args:
            t: Time tensor
            y: State tensor  
            dy: Derivative tensor (ground truth)
            
        Returns:
            Total loss and loss components dictionary
        """
        # Data loss
        data_loss = self.data_loss(t, y, dy)
        
        # Physics loss
        physics_loss = self.physics_loss(t, y)
        
        # Boundary condition loss
        bc_loss = self.boundary_condition_loss(t, y)
        
        # Total loss
        total_loss = (self.data_weight * data_loss + 
                     self.physics_weight * physics_loss + 
                     self.bc_weight * bc_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss,
            'bc_loss': bc_loss
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step for PINN."""
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Reshape tensors for PINN input
        batch_size, seq_len, state_dim = y.shape
        t_flat = t.reshape(-1, 1)  # (batch_size * seq_len, 1)
        y_flat = y.reshape(-1, state_dim)  # (batch_size * seq_len, 2)
        dy_flat = dy.reshape(-1, state_dim)  # (batch_size * seq_len, 2)
        
        # Compute total loss
        total_loss, loss_dict = self.total_loss(t_flat, y_flat, dy_flat)
        
        # Backward pass
        self.manual_backward(total_loss)
        optimizer.step()
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, prog_bar=True)
        
        # Log physics parameters
        self.log('train/omega0', self.omega0, prog_bar=False)
        self.log('train/alpha', self.alpha, prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PINN."""
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Reshape tensors
        batch_size, seq_len, state_dim = y.shape
        t_flat = t.reshape(-1, 1)
        y_flat = y.reshape(-1, state_dim)
        dy_flat = dy.reshape(-1, state_dim)
        
        # Compute losses
        total_loss, loss_dict = self.total_loss(t_flat, y_flat, dy_flat)
        
        # Log validation losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step for PINN."""
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Reshape tensors
        batch_size, seq_len, state_dim = y.shape
        t_flat = t.reshape(-1, 1)
        y_flat = y.reshape(-1, state_dim)
        dy_flat = dy.reshape(-1, state_dim)
        
        # Compute losses
        total_loss, loss_dict = self.total_loss(t_flat, y_flat, dy_flat)
        
        # Get test set name
        test_set_names = ['id', 'id_extra', 'ood_t2', 'ood_t2_extra', 'ood_t3', 'ood_t3_extra']
        test_set = test_set_names[dataloader_idx] if dataloader_idx < len(test_set_names) else f'test_{dataloader_idx}'
        
        # Log test losses
        for key, value in loss_dict.items():
            self.log(f'test/{test_set}/{key}', value, prog_bar=False)
        
        # Compute trajectory prediction for state space evaluation
        with torch.no_grad():
            # Predict trajectory using ODE integration
            from torchdiffeq import odeint
            
            def dynamics_fn(t_val, x_val):
                """Dynamics function for ODE integration."""
                # Ensure input is on correct device
                if not isinstance(t_val, torch.Tensor):
                    t_val = torch.tensor(t_val, dtype=torch.float32, device=self.device)
                if not isinstance(x_val, torch.Tensor):
                    x_val = torch.tensor(x_val, dtype=torch.float32, device=self.device)
                
                # Handle scalar time tensor - ensure it's 1D
                if t_val.dim() == 0:  # Scalar
                    t_val = t_val.unsqueeze(0)  # Make it (1,)
                
                # Prepare network input
                theta = x_val[0:1]  # (1,)
                theta_dot = x_val[1:2]  # (1,)
                network_input = torch.cat([t_val, theta, theta_dot], dim=0).unsqueeze(0)  # (1, 3)
                
                # Get prediction
                pred = self.pinn_net(network_input)
                return pred.squeeze(0)
            
            # Integrate trajectory for first sample in batch
            x0 = y[0, 0, :]  # Initial condition
            # Fix tensor indexing - t is (batch_size, seq_len, 1)
            t_tensor = t[0, :, 0] if t.dim() == 3 else t[0, :]  # Time points
            
            try:
                y_pred = odeint(dynamics_fn, x0, t_tensor, method='dopri5', rtol=1e-6, atol=1e-8)
                y_pred = y_pred.permute(1, 0)  # (time, state) -> (state, time)
                
                # Compute state space MSE
                y_true_sample = y[0]  # (time, state)
                state_mse = F.mse_loss(y_pred, y_true_sample)
                
                # Compute normalized state MSE
                y_norm = torch.norm(y_true_sample)
                normalized_state_mse = state_mse / (y_norm**2 + 1e-8)
                
                self.log(f'test/{test_set}/state_mse', state_mse, prog_bar=False)
                self.log(f'test/{test_set}/normalized_state_mse', normalized_state_mse, prog_bar=False)
                
            except Exception as e:
                print(f"Warning: ODE integration failed in test step: {e}")
                # Set dummy values
                self.log(f'test/{test_set}/state_mse', torch.tensor(1.0), prog_bar=False)
                self.log(f'test/{test_set}/normalized_state_mse', torch.tensor(1.0), prog_bar=False)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",
                "frequency": 1,
                "interval": "epoch"
            }
        }
    
    def predict_derivative(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Predict derivatives using the trained PINN.
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            y: State tensor of shape (batch_size, 2)
            
        Returns:
            Predicted derivatives of shape (batch_size, 2)
        """
        self.eval()
        with torch.no_grad():
            theta = y[:, 0:1]
            theta_dot = y[:, 1:2]
            network_input = torch.cat([t, theta, theta_dot], dim=1)
            pred = self.pinn_net(network_input)
        return pred

