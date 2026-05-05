import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

class MLP(nn.Module):
    def __init__(self, state_c, hidden, num_layers=3, out_c: Optional[int] = None):
        super().__init__()
        self.state_c = state_c
        # Output should be in state space
        self.out_c = out_c if out_c is not None else state_c
        
        # Replace BatchNorm with LayerNorm which works with single samples
        layers = [nn.Linear(state_c, hidden)]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden)
            ])
        layers.extend([
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.out_c)  # Output in state space
        ])
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Reshape input to process each state independently
        original_shape = x.shape
        if x.dim() > 2:
            # Flatten all dimensions except the last (feature dimension)
            x = x.reshape(-1, original_shape[-1])
        
        # Process through network
        out = self.net(x)
        
        # Restore original batch/time dimensions
        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], -1)
            
        return out


class PhysicalModel(nn.Module):
    def __init__(self, feature_library, device, cfg, n_states=None):
        super().__init__()
        self.feature_library = feature_library
        self.cfg = cfg

        # Handle device configuration
        if isinstance(device, str):
            if device == 'gpu':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)
        else:
            self.device = device

        # Number of input features comes from the symbolic library; number
        # of output states is the number of state variables in the system
        # and is supplied by the caller (orthoreg.setup.setup_model derives
        # it from the training-data shape).
        n_features = feature_library.n_output_features_
        if n_states is None:
            # Backwards-compatible fallback for older call sites.
            n_states = cfg.dataset.forecaster.out_dim
        
        # Initialize coefficients with small random values (Kaiming uniform)
        self.coef_ = nn.Parameter(
            torch.empty(
                n_features,  # Input features
                n_states,    # Output states
                device=self.device
            )
        )
        nn.init.kaiming_uniform_(self.coef_, a=np.sqrt(5)) # Initialize like nn.Linear
        
        # Initialize feature selection mask
        self.mask_ = nn.Parameter(
            torch.ones_like(self.coef_),
            requires_grad=False
        )
        
        # Store feature names for printing
        self.feature_names = feature_library.get_feature_names()

    def forward(self, x, fixed_mask=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Apply feature selection mask
        if fixed_mask:
            # During retraining with fixed mask:
            # 1. Zero out coefficients where mask is 0
            # 2. Allow gradients only where mask is 1
            coef_eff = self.coef_ * self.mask_
            # Detach masked coefficients to prevent gradient flow
            coef_eff = torch.where(self.mask_ > 0, coef_eff, coef_eff.detach())
        else:
            coef_eff = self.coef_ * self.mask_

        # Reshape input if needed
        original_shape = x.shape
        if x.dim() > 2:
            # Flatten all dimensions except the last (feature dimension)
            x = x.reshape(-1, original_shape[-1])

        # Compute derivative prediction
        # x shape: (batch_size, n_features)
        # coef_eff shape: (n_features, n_states)
        # We want to multiply x with coef_eff to get (batch_size, n_states)
        out_phy = torch.matmul(x, coef_eff)
        
        # Restore original batch dimensions if needed
        if len(original_shape) > 2:
            out_phy = out_phy.reshape(*original_shape[:-1], -1)
        
        return out_phy

    def threshold_coefficients(self, threshold):
        """Apply thresholding to coefficients like STLSQ"""
        with torch.no_grad():
            # Update mask based on coefficient magnitudes
            self.mask_.data = (torch.abs(self.coef_) >= threshold).float()
            
            # Zero out small coefficients
            self.coef_.data = self.coef_ * self.mask_

    def get_coefficients(self):
        """Get current coefficients with mask applied"""
        with torch.no_grad():
            return self.coef_ * self.mask_

    def print(self):
        """Print equations like PySINDy"""
        coef = self.get_coefficients().cpu().numpy()
        for i in range(coef.shape[1]):  # Loop over output features
            terms = []
            for j in range(coef.shape[0]):  # Loop over input features
                if abs(coef[j, i]) > 1e-10:
                    terms.append(f"{coef[j, i]:.3f} {self.feature_names[j]}")
            if terms:
                print(f"x{i}' = {' + '.join(terms)}")
            else:
                print(f"x{i}' = 0")
