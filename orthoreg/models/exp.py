import lightning as L
import torch
from orthoreg.setup import get_experiment_and_checkpoint_dir
from orthoreg.regularization.orthoreg import l2_penalty, orthoreg_penalty
import numpy as np
from torchdiffeq import odeint
import wandb

class HybridExperiment(L.LightningModule):
    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.automatic_optimization = False
        self.get_dirs()
        self.threshold = cfg.training.symbolic_threshold  # Simple fixed threshold
        
        # Simplified regularization parameters - use config values directly
        if self.cfg.training.regularization == 'l2':
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', cfg.training.l2_symbolic_reg_weight)
            self.node_reg_weight = cfg.training.l2_node_reg_weight
        elif self.cfg.training.regularization == 'orthogonal':
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', cfg.training.orthogonal_symbolic_reg_weight)
            self.node_reg_weight = cfg.training.orthogonal_node_reg_weight
        else:
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', 0.0)
            self.node_reg_weight = 0.0
        
        # Neural network freezing settings
        self.freeze_neural = cfg.training.get('freeze_neural', False)
        self.freeze_epochs = cfg.training.get('freeze_epochs', 0)
        if self.freeze_neural:
            print(f"Neural network will be frozen for the first {self.freeze_epochs} epochs")
        
        # Training stage flags
        self.first_stage = True  # Start with structure discovery
        self.structure_fixed = False  # Will be set to True after first stage

    def get_dirs(self):
        self.experiment_dir, self.checkpoint_dir = get_experiment_and_checkpoint_dir(self.cfg)
    
    @property
    def device(self):
        """Get the current device for the experiment."""
        # Try to get device from the model first
        if hasattr(self, 'net') and hasattr(self.net, 'model_phy'):
            try:
                device = next(self.net.model_phy.parameters()).device
                return device
            except:
                pass
        
        # Fallback to Lightning's device property
        if hasattr(super(), 'device'):
            try:
                device = super().device
                return device
            except:
                pass
        
        # Final fallback to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def get_symbolic_basis_predictions(self, x):
        """
        Get predictions for each active symbolic basis function individually.
        
        Args:
            x: Input tensor of shape [B, T, n_features] where:
                - B is batch size
                - T is sequence length
                - n_features is number of symbolic features
        
        Returns:
            symbolic_basis: Tensor of shape [B, T, n_features, state_dim]
        """
        # Get the symbolic coefficients (already learned)
        coef = self.net.model_phy.coef_  # Shape: [n_features, state_dim]
        
        # Create basis function predictions
        # Each basis function is just the input feature multiplied by its coefficient
        symbolic_basis = x.unsqueeze(-1) * coef.unsqueeze(0).unsqueeze(0)  # [B, T, n_features, state_dim]
        
        return symbolic_basis
    
    def ensure_model_on_device(self, device=None):
        """Ensure the model is on the specified device."""
        if device is None:
            device = self.device
        
        if hasattr(self, 'net'):
            self.net = self.net.to(device)
        
        return device

    def derivative_forward(self, transformed_y, fixed_mask=False):
        """Forward pass for derivative prediction."""
        # Ensure we're on the correct device
        target_device = self.device
        if hasattr(self, 'net'):
            self.net = self.net.to(target_device)
        
        use_fixed_mask = fixed_mask or self.structure_fixed
        out_phy = self.net.model_phy(transformed_y, fixed_mask=use_fixed_mask)
        out_aug = self.net.model_aug(transformed_y) if self.cfg.model.hybrid_setting else None
        
        # Ensure outputs are on the correct device
        out_phy = out_phy.to(target_device)
        if out_aug is not None:
            out_aug = out_aug.to(target_device)
        
        return out_phy, out_aug

    def get_derivative_loss(self, transformed_y, dy, hybrid_setting):
        """Compute normalized MSE loss in derivative space with function space regularization."""
        # Get predictions
        pred_phy, pred_aug = self.derivative_forward(transformed_y)
        
        # Combine predictions if in hybrid mode
        if hybrid_setting:
            if self.cfg.training.regularization == 'orthogonal':
                # Get orthogonal component in function space
                # Since we removed the function_space_regularization, use orthogonal residual instead
                orth_residual = self.orthogonal_residual(pred_aug, pred_phy)
                # Use only the orthogonal component of neural prediction
                pred_dy = pred_phy + orth_residual
            else:
                # Original point-wise orthogonal or L2 regularization
                pred_dy = pred_phy + pred_aug
        else:
            pred_dy = pred_phy
        
        # Use standard SINDy loss (unnormalized MSE) for stronger learning signals
        phy_loss = ((dy - pred_phy)).pow(2).mean()
        aug_loss = torch.tensor(0.0, device=self.device)
        
        if hybrid_setting and pred_aug is not None:
            if self.cfg.training.regularization == 'orthogonal':
                # Empirical-orthogonality penalty: see
                # orthoreg.regularization.orthoreg.orthoreg_penalty.
                symbolic_basis = self.get_symbolic_basis_predictions(transformed_y)
                aug_loss = orthoreg_penalty(pred_aug, symbolic_basis)
            else:
                aug_loss = l2_penalty(pred_aug)
        
        # Use unnormalized MSE for total loss (SINDy-style)
        total_loss = ((dy - pred_dy)).pow(2).mean()
        
        return {
            'total_loss': total_loss,
            'phy_loss': phy_loss,
            'aug_loss': aug_loss,
            'pred_dy': pred_dy
        } 

    def configure_optimizers(self):
        # Single learning rate for all components
        params_to_optimize = [
            {'params': [self.net.model_phy.coef_]},
            {'params': [p for n, p in self.net.model_aug.named_parameters()]}
        ] if self.cfg.model.hybrid_setting else [
            {'params': [self.net.model_phy.coef_]}
        ]
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.cfg.training.lr)
        
        # Use training loss for scheduling instead of validation loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",  # Use training loss instead of validation
                "frequency": 1,
                "interval": "epoch"
            }
        }

    def get_symbolic_basis(self, x):
        """Get symbolic basis functions at given points."""
        return self.net.model_phy.get_features(x)
        
    def orthogonal_residual(self, pred_aug, pred_phy):
        # pred_aug: (batch, d)
        # pred_phy: (batch, d)
        dot = torch.sum(pred_aug * pred_phy, dim=-1, keepdim=True)  # (batch, 1)
        norm_sq = torch.sum(pred_phy ** 2, dim=-1, keepdim=True) + 1e-8
        proj = (dot / norm_sq) * pred_phy  # projection onto pred_phy
        orth_residual = pred_aug - proj
        return orth_residual

    def update_training_phase(self):
        """Simple phase transition with better logging"""
        if self.structure_fixed:
            # Already in second stage, no changes needed
            return
            
        # Get current epoch
        current_epoch = self.current_epoch
        reg_decay_epochs = self.cfg.training.n_derivative_epochs
        
        if current_epoch >= reg_decay_epochs and self.first_stage:
            # Transition to second stage at the specified epoch
            print(f"\n{'='*60}")
            print(f"=== TRANSITIONING TO SECOND STAGE AT EPOCH {current_epoch} ===")
            print(f"=== Freezing symbolic structure and retraining parameters ===")
            print(f"{'='*60}")
            
            self.first_stage = False
            self.structure_fixed = True
            
            # Store the feature mask based on current thresholding
            with torch.no_grad():
                self.feature_mask = (torch.abs(self.net.model_phy.coef_) > self.threshold).float().detach().cpu().numpy()
                num_active_features = np.sum(self.feature_mask)
                print(f"Active features after thresholding: {num_active_features}")
                
            # Reset optimizer for second stage
            self.trainer.optimizers[0] = self.configure_optimizers()["optimizer"]
            print(f"Optimizer reset for second stage training")
            print(f"{'='*60}\n")

    def training_step(self, batch, batch_idx):
        # Update regularization weights based on current epoch
        self.update_training_phase()
        
        # If we've transitioned to the second stage, use retrain logic
        if self.structure_fixed:
            return self.retrain_parameters(batch)
        
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Unpack batch data
        y, transformed_y, dy, t = batch
        
        # Forward pass and compute base MSE losses
        losses = self.get_derivative_loss(transformed_y, dy, self.cfg.model.hybrid_setting)
        
        # L1 regularization for physical coefficients (sparsity)
        l1_reg = torch.norm(self.net.model_phy.coef_, p=1) * self.l1_reg_weight
        
        # NODE Output Regularization
        node_reg = torch.tensor(0.0, device=self.device)
        if self.cfg.model.hybrid_setting:
            pred_phy, pred_aug = self.derivative_forward(transformed_y)
            if pred_aug is not None:
                if self.cfg.training.regularization == 'l2':
                    node_reg = torch.mean(torch.norm(pred_aug, p=2, dim=-1)) * self.node_reg_weight
                elif self.cfg.training.regularization == 'orthogonal':
                    orth_residual = self.orthogonal_residual(pred_aug, pred_phy)
                    node_reg = torch.mean(torch.norm(orth_residual, p=2, dim=-1)) * self.node_reg_weight
        
        # Compute total loss and backward
        physical_loss_reg = losses['phy_loss'] + l1_reg
        
        # During initial freezing period, only train the symbolic part if freezing is enabled
        freeze_neural = self.cfg.training.get('freeze_neural', False)
        freeze_epochs = self.cfg.training.get('freeze_epochs', 0)
        is_frozen = freeze_neural and self.current_epoch < freeze_epochs
        
        if is_frozen:
            # Only use symbolic loss during freezing period
            total_loss_backward = physical_loss_reg
            self.log('train/neural_frozen', 1.0, prog_bar=True)
        else:
            # Add neural component loss after freezing period
            # Note: node_reg already includes aug_loss, don't add it twice
            total_loss_backward = physical_loss_reg + node_reg
            if self.cfg.model.hybrid_setting:
                self.log('train/neural_frozen', 0.0, prog_bar=True)
            
        self.manual_backward(total_loss_backward)
        
        # If neural network is frozen, zero out its gradients
        if is_frozen and self.cfg.model.hybrid_setting:
            for param in self.net.model_aug.parameters():
                param.grad = None
                
        self.clip_gradients(optimizer, gradient_clip_val=self.cfg.training.gradient_clip_value, gradient_clip_algorithm="norm")
        optimizer.step()

        # Log losses with consistent keys
        prefix = 'retrain' if self.structure_fixed else 'train'
        self.log(f'{prefix}/total_loss', losses['total_loss'], prog_bar=True)
        self.log(f'{prefix}/phy_loss', losses['phy_loss'], prog_bar=True)
        self.log(f'{prefix}/aug_loss', losses['aug_loss'], prog_bar=True)
        self.log(f'{prefix}/l1_reg', l1_reg, prog_bar=False)
        self.log(f'{prefix}/node_reg', node_reg, prog_bar=False)
        self.log(f'{prefix}/total_loss_reg', total_loss_backward, prog_bar=True)
        self.log(f'{prefix}/stage', 1.0 if self.first_stage else 2.0, prog_bar=True)
        
        return total_loss_backward

    # Add this method to manually step epoch-based schedulers
    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers() # Get the scheduler
        if scheduler:
            # Use consistent logging keys for scheduler monitoring
            prefix = 'retrain' if self.structure_fixed else 'train'
            metric_val = self.trainer.callback_metrics.get(f"{prefix}/total_loss")
            if metric_val is not None:
                # Step scheduler with the metric monitored
                scheduler.step(metric_val)
                
    def on_validation_epoch_end(self):
        """Handle validation epoch end operations."""
        scheduler = self.lr_schedulers() # Get the scheduler
        if scheduler:
            # Use consistent logging keys for scheduler monitoring
            prefix = 'retrain' if self.structure_fixed else 'train'
            metric_val = self.trainer.callback_metrics.get(f"{prefix}/total_loss")
            if metric_val is not None:
                # Step scheduler with the metric monitored
                scheduler.step(metric_val)

    def retrain_parameters(self, batch):
        """Retrain parameters while keeping symbolic structure fixed."""
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Unpack batch data
        y, transformed_y, dy, t = batch
        
        # Forward pass with fixed mask
        losses = self.get_derivative_loss(transformed_y, dy, self.cfg.model.hybrid_setting)
        
        # Only compute loss for parameter optimization (no regularization)
        total_loss_backward = losses['total_loss']
            
        self.manual_backward(total_loss_backward)
        
        # Ensure zero gradients for masked coefficients using stored feature mask
        with torch.no_grad():
            mask_tensor = torch.tensor(self.feature_mask, dtype=torch.float32, device=self.device)
            self.net.model_phy.coef_.grad *= mask_tensor
        
        optimizer.step()
        
        # Ensure coefficients remain zero where masked
        with torch.no_grad():
            self.net.model_phy.coef_.data *= mask_tensor

        # Log losses with consistent keys (same as training_step)
        prefix = 'retrain' if self.structure_fixed else 'train'
        self.log(f'{prefix}/total_loss', losses['total_loss'], prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/phy_loss', losses['phy_loss'], prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/aug_loss', losses['aug_loss'], prog_bar=True, on_epoch=True, on_step=False)
        self.log(f'{prefix}/stage', 2.0, prog_bar=True, on_epoch=True, on_step=False)
        
        return total_loss_backward

    def validation_step(self, batch, batch_idx):
        y, transformed_y, dy, t = batch
        
        # Compute base losses (no grad needed here, but derivative_forward handles it)
        losses = self.get_derivative_loss(transformed_y, dy, self.cfg.model.hybrid_setting)
        
        # Calculate L1 regularization for physical coefficients
        l1_reg = torch.norm(self.net.model_phy.coef_, p=1) * self.l1_reg_weight
        
        # Total validation loss including regularization (for scheduler and logging)
        total_loss_reg = losses['phy_loss'] + l1_reg
        if self.cfg.model.hybrid_setting:
            total_loss_reg += losses['aug_loss']

        # Use consistent logging keys
        prefix = 'retrain' if self.structure_fixed else 'train'
        self.log(f'val/{prefix}_total_loss', losses['total_loss'])
        self.log(f'val/{prefix}_phy_loss', losses['phy_loss'])
        self.log(f'val/{prefix}_aug_loss', losses['aug_loss'])
        self.log(f'val/{prefix}_l1_reg', l1_reg)
        self.log(f'val/{prefix}_total_loss_reg', total_loss_reg) # Log total loss with reg
        
        return total_loss_reg # Return loss used for scheduler
        
    def compute_dynamics_metrics(self, pred_coeffs, true_coeffs_dict, feature_names, batch_idx=0, dataloader_idx=0):
        """Compute metrics comparing predicted dynamics to true dynamics."""
        # Convert predictions and true coefficients to numpy arrays
        pred = pred_coeffs.detach().cpu().numpy()
        true = np.zeros_like(pred)
        for i, feature in enumerate(feature_names):
            if feature in true_coeffs_dict:
                true[i] = true_coeffs_dict[feature].cpu().numpy()
        
        # Compute core metrics
        coef_l2 = np.linalg.norm(pred - true)
        coef_l2_norm = coef_l2 / (np.linalg.norm(true) + 1e-8)
        
        # Structure metrics
        true_nonzero = np.abs(true) > self.threshold
        pred_nonzero = np.abs(pred) > self.threshold
        true_pos = np.sum(true_nonzero & pred_nonzero)
        false_pos = np.sum(~true_nonzero & pred_nonzero)
        false_neg = np.sum(true_nonzero & ~pred_nonzero)
        
        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Only print for first batch of first test loader
        if batch_idx == 0 and dataloader_idx == 0:
            # Get significant terms
            sig_terms = [(i, feature) for i, feature in enumerate(feature_names)
                        if (np.abs(true[i]) > self.threshold).any() or 
                           (np.abs(pred[i]) > self.threshold).any()]
            
            print("\n=== Symbolic Regression Quality ===")
            print(f"L2: {coef_l2:.3f} (norm: {coef_l2_norm:.3f})")
            print(f"Structure: F1={f1:.3f} (P={precision:.3f}, R={recall:.3f})")
            
            if sig_terms:
                print("\nCoefficients:")
                for idx, feature in sig_terms:
                    if np.any(np.abs(true[idx]) > self.threshold) or np.any(np.abs(pred[idx]) > self.threshold):
                        print(f"{feature:<10}: {true[idx]} -> {pred[idx]}")
        
        # Return only core metrics - theoretical metrics are already logged in test_step
        metrics = {
            'coef_l2_dist': coef_l2,
            'coef_l2_norm': coef_l2_norm,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        # Wandb log metrics
        self.log('test/coef_l2_dist', coef_l2, prog_bar=False, on_epoch=True, on_step=False)
        self.log('test/coef_l2_norm', coef_l2_norm, prog_bar=False, on_epoch=True, on_step=False)
        self.log('test/f1', f1, prog_bar=False, on_epoch=True, on_step=False)
        self.log('test/precision', precision, prog_bar=False, on_epoch=True, on_step=False)
        self.log('test/recall', recall, prog_bar=False, on_epoch=True, on_step=False)
        
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Enhanced test step with better numerical stability."""
        y, transformed_y, dy, t = batch
        
        # Get test set name robustly from datamodule
        if hasattr(self.trainer.datamodule, '_test_set_names'):
            test_set_key = self.trainer.datamodule._test_set_names[dataloader_idx]
        else:
            # Fallback to old order if attribute missing
            test_set_keys = [
                'id_test_data', 
                'id_test_data_extra',
                'ood_test_data_t2',
                'ood_test_data_t2_extra',
                'ood_test_data_t3', 
                'ood_test_data_t3_extra'
            ]
            test_set_key = test_set_keys[dataloader_idx]
        current_test_dataset = self.trainer.datamodule.data_dict.get(test_set_key, None)
        test_set = test_set_key.replace('_test_data', '').replace('_data', '')
        
        # Compute derivative space losses
        losses = self.get_derivative_loss(transformed_y, dy, self.cfg.model.hybrid_setting)
        l1_reg = torch.norm(self.net.model_phy.coef_, p=1) * self.l1_reg_weight
        total_loss = losses['phy_loss'] + l1_reg + losses['aug_loss']
        
        # Compute normalized derivative MSE (scale-invariant)
        dy_norm = torch.norm(dy, dim=-1).mean()
        normalized_derivative_mse = losses['total_loss'] / (dy_norm ** 2 + 1e-8)
        
        # Compute state space trajectory and MSE
        with torch.no_grad():
            # Get feature library from datamodule
            feature_library = self.trainer.datamodule.feature_library
            
            def dynamics(t, x):
                # Transform state to feature space
                x_features = feature_library.transform(x.cpu().numpy())
                x_features = torch.tensor(x_features, dtype=torch.float32, device=x.device)
                
                # Get predictions from both physical and neural components
                pred_phy, pred_aug = self.derivative_forward(x_features)
                
                # Combine predictions if in hybrid mode
                if self.cfg.model.hybrid_setting:
                    pred_dy = pred_phy + pred_aug
                else:
                    pred_dy = pred_phy
                    
                return pred_dy
            
            
            # Get initial conditions
            x0 = y[:, 0, :]  # Shape: (batch_size, state_dim)
            
            # Enhanced integration stability
            try:
                # Try dopri5 with tighter tolerances first
                trajectories = odeint(
                    dynamics, 
                    x0.to(self.device), 
                    t[0].to(self.device),
                    # method='euler',
                    # rtol=1e-3,
                    # atol=1e-4,
                    rtol=1e-5,  # Tighter relative tolerance
                    atol=1e-7,  # Tighter absolute tolerance
                    method='dopri5',
                    options={'max_num_steps': 2000}  # Allow more integration steps
                )
            except Exception as e:
                print(f"Warning: euler failed ({e}), reducing tolerances")
                # Fall back to RK4 with smaller step size
                trajectories = odeint(
                    dynamics, 
                    x0.to(self.device), 
                    t[0].to(self.device),
                    method='euler',
                    rtol=1e-2,
                    atol=1e-3,
                    options={'step_size': 0.01}  # Smaller step size for stability
                )
            
            y_pred = trajectories.permute(1, 0, 2)  # (batch, time, state)
            
            # Compute MSE without normalization
            state_mse = torch.mean((y - y_pred) ** 2)
            
            # Compute normalized state MSE (scale-invariant)
            y_norm = torch.norm(y, dim=-1).mean()
            normalized_state_mse = state_mse / (y_norm ** 2 + 1e-8)
        
        # Print MSE for each test case (only once per test set)
        if batch_idx == 0:
            print(f"\n{test_set} Results:")
            print(f"Derivative Space MSE: {losses['total_loss']:.3e}")
            print(f"Normalized Derivative MSE: {normalized_derivative_mse:.3e}")
            print(f"State Space MSE: {state_mse:.3e}")
            print(f"Normalized State MSE: {normalized_state_mse:.3e}")
            
            # Log both derivative and state space metrics
            if wandb.run is not None:
                metrics = {
                    f'test/{test_set}/derivative_mse/dataloader_idx_{dataloader_idx}': losses['total_loss'],
                    f'test/{test_set}/normalized_derivative_mse/dataloader_idx_{dataloader_idx}': normalized_derivative_mse,
                    f'test/{test_set}/state_mse/dataloader_idx_{dataloader_idx}': state_mse,
                    f'test/{test_set}/normalized_state_mse/dataloader_idx_{dataloader_idx}': normalized_state_mse,
                    f'test/{test_set}/phy_loss/dataloader_idx_{dataloader_idx}': losses['phy_loss'],
                    f'test/{test_set}/aug_loss/dataloader_idx_{dataloader_idx}': losses['aug_loss'],
                    f'test/{test_set}/l1_reg/dataloader_idx_{dataloader_idx}': l1_reg
                }
                wandb.log(metrics)
            
            # Also log to Lightning for terminal output
            self.log(f'test/{test_set}/derivative_mse', losses['total_loss'], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/normalized_derivative_mse', normalized_derivative_mse, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/state_mse', state_mse, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/normalized_state_mse', normalized_state_mse, prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/phy_loss', losses['phy_loss'], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/aug_loss', losses['aug_loss'], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f'test/{test_set}/l1_reg', l1_reg, prog_bar=False, on_epoch=True, on_step=False)
        
        # Coefficient monitoring
        num_nonzero = torch.sum(torch.abs(self.net.model_phy.coef_) > self.threshold)
        self.log(f'test/{test_set}/num_nonzero_terms', num_nonzero, prog_bar=False, on_epoch=True, on_step=False)  
        top_k_values, _ = torch.topk(torch.abs(self.net.model_phy.coef_.flatten()), k=5)
        # Log individual top-k values instead of the whole tensor
        for i, val in enumerate(top_k_values):
            self.log(f'test/{test_set}/top_{i+1}_value', val, prog_bar=False, on_epoch=True, on_step=False)
        # Log max value for easy access
        self.log(f'test/{test_set}/max_coef_value', top_k_values[0], prog_bar=False, on_epoch=True, on_step=False)
        
        # Neural contribution
        if self.cfg.model.hybrid_setting:
            pred_phy, pred_aug = self.derivative_forward(transformed_y)
            neural_ratio = torch.norm(pred_aug) / (torch.norm(pred_phy) + 1e-8)
            
            # Compare residuals with ground truth
            residual = dy - pred_phy
            residual_norm = torch.norm(residual)
            dy_norm = torch.norm(dy)
            unexplained_ratio = residual_norm/dy_norm

            # Dot product between phy and aug components should be near zero
            dot_product = torch.sum(pred_phy * pred_aug) / (torch.norm(pred_phy) * torch.norm(pred_aug) + 1e-8)
            orthogonality = torch.abs(dot_product)

            # Log to both wandb and Lightning
            if wandb.run is not None:
                metrics = {
                    f'test/{test_set}/neural_ratio/dataloader_idx_{dataloader_idx}': neural_ratio,
                    f'test/{test_set}/unexplained_ratio/dataloader_idx_{dataloader_idx}': unexplained_ratio,
                    f'test/{test_set}/orthogonality/dataloader_idx_{dataloader_idx}': orthogonality
                }
                wandb.log(metrics)

            # Also log to Lightning for terminal output
            self.log(f'test/{test_set}/neural_ratio', neural_ratio)
            self.log(f'test/{test_set}/unexplained_ratio', unexplained_ratio)
            self.log(f'test/{test_set}/orthogonality', orthogonality)
        

        # Compute dynamics metrics if available
        if current_test_dataset is not None and hasattr(current_test_dataset, 'W_true'):
            metrics = self.compute_dynamics_metrics(
                self.net.model_phy.get_coefficients(),
                current_test_dataset.W_true,
                self.net.model_phy.feature_names,
                batch_idx,
                dataloader_idx
            )
            # Only log essential metrics
            for name, value in metrics.items():
                self.log(f'test/{test_set}/{name}', value, prog_bar=False, on_epoch=True, on_step=False)
        
        return total_loss

    def compare_model_parameters(self, true_params):
        """Compare model parameters with true parameters and print sparsity."""
        model_params = self.net.model_phy.coef_.detach().cpu().numpy()
        sparsity = np.sum(model_params != 0) / model_params.size
        true_sparsity = np.sum(true_params != 0) / true_params.size

        print("Model Sparsity:", sparsity)
        print("True Sparsity:", true_sparsity)

        # Compare parameters
        param_diff = np.abs(model_params - true_params)
        print("Parameter Differences:", param_diff)