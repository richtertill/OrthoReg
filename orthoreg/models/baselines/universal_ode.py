"""
Universal Ordinary Differential Equations (UODEs) implementation for pendulum dynamics.

Based on Chen et al. (2020): "Universal Differential Equations for Scientific Machine Learning"
and Rackauckas et al. (2021): "Universal Differential Equations for Scientific Machine Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
import lightning as L
from torchdiffeq import odeint


class UniversalODENetwork(nn.Module):
    """
    Universal ODE network that learns unknown dynamics components.
    
    The network learns a function f(t, x) where x is the state vector and f represents
    the unknown dynamics that need to be learned from data.
    """
    
    def __init__(self, state_dim: int = 2, hidden_dim: int = 64, num_layers: int = 3,
                 activation: str = 'tanh', include_time: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.include_time = include_time
        
        # Input dimension: state_dim + 1 (for time) if include_time else state_dim
        input_dim = state_dim + 1 if include_time else state_dim
        
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
            nn.Linear(hidden_dim, state_dim)
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
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            t: Time tensor of shape (batch_size,) or scalar
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Output tensor of shape (batch_size, state_dim) representing dx/dt
        """
        # Debug logging (commented out for production)
        # print(f"[UniversalODENetwork.forward] Input shapes: t={t.shape}, x={x.shape}")
        # print(f"[UniversalODENetwork.forward] Input dims: t.dim()={t.dim()}, x.dim()={x.dim()}")
        
        if self.include_time:
            # Handle time tensor - ensure it's 1D with batch_size elements
            if t.dim() == 0:  # Scalar time (from ODE integration)
                # For scalar time, expand to match x's batch dimension
                if x.dim() == 1:
                    # x is 1D (state_dim), make it 2D (1, state_dim) and expand t
                    x = x.unsqueeze(0)  # (1, state_dim)
                    t = t.unsqueeze(0)  # (1,)
                else:
                    # x is already 2D (batch_size, state_dim)
                    t = t.expand(x.shape[0])  # (batch_size,)
            elif t.dim() == 1:  # 1D time tensor
                if t.shape[0] == 1:  # Single time value
                    t = t.expand(x.shape[0])
                elif t.shape[0] != x.shape[0]:  # Mismatched batch sizes
                    t = t.expand(x.shape[0])
            elif t.dim() == 2:  # 2D time tensor (batch_size, 1)
                t = t.squeeze(1)  # Remove the extra dimension
                if t.shape[0] != x.shape[0]:
                    t = t.expand(x.shape[0])
            elif t.dim() == 3:  # 3D time tensor (batch_size, seq_len, 1)
                t = t[:, 0, 0]  # Take first time step from first sequence
                if t.shape[0] != x.shape[0]:
                    t = t.expand(x.shape[0])
            
            # Ensure time has the same batch size as state
            if t.shape[0] != x.shape[0]:
                t = t.expand(x.shape[0])
            
            # Handle state tensor - ensure it's 2D (batch_size, state_dim)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Make x 2D: (1, state_dim)
            elif x.dim() == 3:  # 3D tensor (batch_size, seq_len, state_dim)
                x = x[:, 0, :]  # Take first time step: (batch_size, state_dim)
            
            # print(f"[UniversalODENetwork.forward] After reshaping: t={t.shape}, x={x.shape}")
            
            # Now both t and x should be 2D: (batch_size, 1) and (batch_size, state_dim)
            input_tensor = torch.cat([t.unsqueeze(1), x], dim=1)
        else:
            # Handle state tensor - ensure it's 2D (batch_size, state_dim)
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Make x 2D: (1, state_dim)
            elif x.dim() == 3:  # 3D tensor (batch_size, seq_len, state_dim)
                x = x[:, 0, :]  # Take first time step: (batch_size, state_dim)
            
            input_tensor = x
        
        # print(f"[UniversalODENetwork.forward] Final input_tensor shape: {input_tensor.shape}")
        return self.network(input_tensor)


class UniversalODEExperiment(L.LightningModule):
    """
    Lightning module for training Universal ODE on pendulum dynamics.
    """
    
    def __init__(self, cfg, feature_library=None):
        super().__init__()
        self.cfg = cfg
        self.feature_library = feature_library
        
        state_dim = 2  # (theta, theta_dot)
        hidden_dim = cfg.model.get('hidden_dim', 64)
        num_layers = cfg.model.get('num_layers', 3)
        activation = cfg.model.get('activation', 'tanh')
        include_time = cfg.model.get('include_time', True)
        
        # Initialize Universal ODE network
        self.uode_net = UniversalODENetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            include_time=include_time
        )
        
        # Known physics parameters (can be learnable or fixed)
        self.omega0 = nn.Parameter(torch.tensor(1.0))  # Natural frequency
        self.alpha = nn.Parameter(torch.tensor(0.2))   # Damping coefficient
        
        # Loss weights
        self.data_weight = cfg.training.get('data_weight', 1.0)
        self.physics_weight = cfg.training.get('physics_weight', 0.1)
        
        # Training parameters
        self.lr = cfg.training.lr
        self.automatic_optimization = False
        
        # ODE integration parameters
        self.ode_method = cfg.training.get('ode_method', 'dopri5')
        self.ode_rtol = cfg.training.get('ode_rtol', 1e-6)
        self.ode_atol = cfg.training.get('ode_atol', 1e-8)
        
    def known_dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Known damped-pendulum dynamics.

        theta_dot = theta_dot (identity)
        theta_ddot = -omega_0^2 * sin(theta) - alpha * theta_dot

        Args:
            t: time tensor.
            x: state tensor of shape (batch, 2) = (theta, theta_dot).

        Returns:
            tensor of shape (batch, 2) with the known dynamics.
        """
        # Handle both 1D and 2D x tensors
        if x.dim() == 1:
            theta = x[0:1].unsqueeze(0)  # (1, 1)
            theta_dot = x[1:2].unsqueeze(0)  # (1, 1)
        else:
            theta = x[:, 0:1]
            theta_dot = x[:, 1:2]
        
        # Known dynamics
        dtheta_dt = theta_dot
        dtheta_dot_dt = -self.omega0**2 * torch.sin(theta) - self.alpha * theta_dot
        
        return torch.cat([dtheta_dt, dtheta_dot_dt], dim=1)
    
    def full_dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Full dynamics combining known physics and learned components.
        
        dx/dt = f_known(t, x) + f_learned(t, x)
        
        Args:
            t: Time tensor
            x: State tensor
            
        Returns:
            Full dynamics tensor
        """
        # Known physics dynamics
        known_dyn = self.known_dynamics(t, x)
        
        # Learned dynamics from Universal ODE network
        learned_dyn = self.uode_net(t, x)
        
        # Combine known and learned dynamics
        full_dyn = known_dyn + learned_dyn
        
        return full_dyn
    
    def data_loss(self, t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss by integrating the learned dynamics.
        
        Args:
            t: Time tensor of shape (batch_size, seq_len)
            y: State tensor of shape (batch_size, seq_len, state_dim)
            dy: Derivative tensor (ground truth) of shape (batch_size, seq_len, state_dim)
            
        Returns:
            Data loss tensor
        """
        batch_size, seq_len, state_dim = y.shape
        
        # Reshape for processing
        t_flat = t.reshape(-1)  # (batch_size * seq_len,)
        y_flat = y.reshape(-1, state_dim)  # (batch_size * seq_len, state_dim)
        dy_flat = dy.reshape(-1, state_dim)  # (batch_size * seq_len, state_dim)
        
        # Get predicted derivatives using full dynamics
        pred_dy = self.full_dynamics(t_flat, y_flat)
        
        # Compute MSE loss
        data_loss = F.mse_loss(pred_dy, dy_flat)
        
        return data_loss
    
    def trajectory_loss(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory loss by integrating the learned dynamics and comparing with true trajectory.
        
        Args:
            t: Time tensor of shape (batch_size, seq_len)
            y: True state tensor of shape (batch_size, seq_len, state_dim)
            
        Returns:
            Trajectory loss tensor
        """
        batch_size, seq_len, state_dim = y.shape
        
        # Integrate trajectories for each sample in the batch
        total_loss = 0.0
        
        for i in range(batch_size):
            # Get initial condition and time points for this sample
            x0 = y[i, 0, :]  # Initial state
            t_sample = t[i, :, 0]  # Time points for this sample
            
            try:
                # Integrate using the full dynamics
                y_pred = odeint(
                    self.full_dynamics,
                    x0,
                    t_sample,
                    method=self.ode_method,
                    rtol=self.ode_rtol,
                    atol=self.ode_atol
                )
                
                # Reshape prediction to match true trajectory
                y_pred = y_pred.permute(1, 0)  # (time, state) -> (state, time)
                y_true = y[i]  # (time, state)
                
                # Compute MSE loss for this trajectory
                traj_loss = F.mse_loss(y_pred, y_true)
                total_loss += traj_loss
                
            except Exception as e:
                print(f"Warning: ODE integration failed for sample {i}: {e}")
                # Use a large penalty for failed integration
                total_loss += torch.tensor(100.0, device=self.device)
        
        return total_loss / batch_size
    
    def physics_regularization(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Physics-based regularization to encourage the learned component to be small
        when the known physics is sufficient.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            Physics regularization loss
        """
        # Get learned dynamics
        learned_dyn = self.uode_net(t, y)
        
        # Regularize the magnitude of learned dynamics
        reg_loss = torch.mean(torch.norm(learned_dyn, p=2, dim=1))
        
        return reg_loss
    
    def total_loss(self, t: torch.Tensor, y: torch.Tensor, dy: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss combining data and physics losses.
        
        Args:
            t: Time tensor
            y: State tensor
            dy: Derivative tensor (ground truth)
            
        Returns:
            Total loss and loss components dictionary
        """
        # Data loss (derivative fitting)
        data_loss = self.data_loss(t, y, dy)
        
        # Trajectory loss (optional, can be computationally expensive)
        trajectory_loss = self.trajectory_loss(t, y) if self.cfg.training.get('use_trajectory_loss', False) else torch.tensor(0.0, device=self.device)
        
        # Physics regularization
        physics_reg = self.physics_regularization(t.reshape(-1), y.reshape(-1, y.shape[-1]))
        
        # Total loss
        total_loss = (self.data_weight * data_loss + 
                     self.cfg.training.get('trajectory_weight', 0.0) * trajectory_loss +
                     self.physics_weight * physics_reg)
        
        loss_dict = {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'trajectory_loss': trajectory_loss,
            'physics_reg': physics_reg
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step for Universal ODE."""
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Compute total loss
        total_loss, loss_dict = self.total_loss(t, y, dy)
        
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
        """Validation step for Universal ODE."""
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Compute losses
        total_loss, loss_dict = self.total_loss(t, y, dy)
        
        # Log validation losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step for Universal ODE."""
        # Unpack batch
        y, transformed_y, dy, t = batch
        
        # Compute losses
        total_loss, loss_dict = self.total_loss(t, y, dy)
        
        # Get test set name
        test_set_names = ['id', 'id_extra', 'ood_t2', 'ood_t2_extra', 'ood_t3', 'ood_t3_extra']
        test_set = test_set_names[dataloader_idx] if dataloader_idx < len(test_set_names) else f'test_{dataloader_idx}'
        
        # Log test losses
        for key, value in loss_dict.items():
            self.log(f'test/{test_set}/{key}', value, prog_bar=False)
        
        # Compute trajectory prediction for state space evaluation
        with torch.no_grad():
            # Predict trajectory using ODE integration
            x0 = y[0, 0, :]  # Initial condition for first sample
            # Fix tensor indexing - handle different t shapes
            if t.dim() == 3:  # (batch_size, seq_len, 1)
                t_sample = t[0, :, 0]  # Time points for first sample
            elif t.dim() == 2:  # (batch_size, seq_len)
                t_sample = t[0, :]  # Time points for first sample
            else:  # (seq_len,)
                t_sample = t  # Time points
            
            try:
                y_pred = odeint(
                    self.full_dynamics,
                    x0,
                    t_sample,
                    method=self.ode_method,
                    rtol=self.ode_rtol,
                    atol=self.ode_atol
                )
                
                y_pred = y_pred.permute(1, 0)  # (time, state) -> (state, time)
                y_true_sample = y[0]  # (time, state)
                
                # Compute state space MSE
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
        Predict derivatives using the trained Universal ODE.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            Predicted derivatives
        """
        self.eval()
        with torch.no_grad():
            pred = self.full_dynamics(t, y)
        return pred
    
    def predict_trajectory(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict full trajectory using the trained Universal ODE.
        
        Args:
            x0: Initial condition tensor of shape (state_dim,)
            t: Time points tensor of shape (seq_len,)
            
        Returns:
            Predicted trajectory tensor of shape (seq_len, state_dim)
        """
        self.eval()
        with torch.no_grad():
            y_pred = odeint(
                self.full_dynamics,
                x0,
                t,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol
            )
            return y_pred.permute(1, 0)  # (time, state) -> (state, time)

