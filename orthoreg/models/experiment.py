import lightning as L
import torch
from orthoreg.feature_library import get_experiment_and_checkpoint_dir
from orthoreg.regularization.orthoreg import l2_penalty, orthoreg_penalty
import numpy as np
from torchdiffeq import odeint
from orthoreg.utils import get_wandb


def _safe_div(numerator, denominator):
    """Ratio that returns 0.0 instead of dividing by zero."""
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def support_metrics(pred, true, threshold, n_out_of_library_terms=0):
    """Support-recovery metrics over the library slots (feature x state dim).

    Parameters
    ----------
    pred, true : np.ndarray, shape ``[n_features, state_dim]``
        Learned and ground-truth coefficients. The true support is every
        nonzero entry of ``true``; the predicted support is every entry of
        ``pred`` whose magnitude exceeds ``threshold``.
    threshold : float
        Magnitude above which a learned coefficient counts as active.
    n_out_of_library_terms : int
        Number ``K`` of true terms that the library cannot express. They
        can never be recovered, so they only enter the full-support scores.

    Returns
    -------
    dict
        ``precision = TP / nnz``, ``recall = TP / P``,
        ``f1 = 2 TP / (nnz + P)``, ``fpr = FP / (#slots - P)``,
        ``full_recall = TP / (P + K)`` and ``full_f1 = 2 TP / (nnz + P + K)``,
        where ``P`` is the number of true in-library slots and ``nnz`` the
        number of predicted active slots. Counts are returned as well.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have the same shape, got {pred.shape} and {true.shape}")

    true_support = np.abs(true) > 0
    pred_support = np.abs(pred) > threshold

    n_slots = true.size
    n_true = int(np.sum(true_support))
    nnz = int(np.sum(pred_support))
    true_pos = int(np.sum(true_support & pred_support))
    false_pos = nnz - true_pos
    false_neg = n_true - true_pos
    K = int(n_out_of_library_terms)

    return {
        'precision': _safe_div(true_pos, nnz),
        'recall': _safe_div(true_pos, n_true),
        'f1': _safe_div(2 * true_pos, nnz + n_true),
        'fpr': _safe_div(false_pos, n_slots - n_true),
        'full_recall': _safe_div(true_pos, n_true + K),
        'full_f1': _safe_div(2 * true_pos, nnz + n_true + K),
        'true_pos': true_pos,
        'false_pos': false_pos,
        'false_neg': false_neg,
        'num_nonzero': nnz,
    }


class HybridExperiment(L.LightningModule):
    """Hybrid SINDy + neural augmentation trained with Algorithm 1.

    The model predicts ``f_hat(x) = sum_j w_j phi_j(x) + f_aug(x)`` and every
    optimization step minimizes

        mean_i ||y_i - f_hat(x_i)||^2 + mu * ||w||_1 + L_reg,

    where ``L_reg`` is selected by ``training.regularization``:

    - ``orthogonal``: ``lambda * sum_{j,k} <f_aug, phi_j e_k>_D^2``,
    - ``l2``: ``lambda * mean_i ||f_aug(x_i)||^2``,
    - ``none``: no penalty (pure symbolic model, no network).
    """

    def __init__(self, net, cfg):
        super().__init__()
        self.net = net
        self.cfg = cfg
        self.automatic_optimization = False
        self.get_dirs()
        self.threshold = cfg.training.symbolic_threshold  # Simple fixed threshold

        # Regularization weights: mu (L1 on the coefficients) and lambda
        # (penalty on the neural augmentation), per method.
        if self.cfg.training.regularization == 'l2':
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', cfg.training.l2_symbolic_reg_weight)
            self.node_reg_weight = cfg.training.l2_node_reg_weight
        elif self.cfg.training.regularization == 'orthogonal':
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', cfg.training.orthogonal_symbolic_reg_weight)
            self.node_reg_weight = cfg.training.orthogonal_node_reg_weight
        else:
            self.l1_reg_weight = cfg.training.get('l1_reg_weight', 0.0)
            self.node_reg_weight = 0.0

        # Number of true terms outside the library (full-support metrics).
        self.n_out_of_library_terms = cfg.dataset.get('n_out_of_library_terms', 0)

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

    def predict_derivative(self, transformed_y):
        """Hybrid prediction ``f_phy + f_aug`` (``f_phy`` alone without a network)."""
        pred_phy, pred_aug = self.derivative_forward(transformed_y)
        pred_dy = pred_phy + pred_aug if pred_aug is not None else pred_phy
        return pred_dy, pred_phy, pred_aug

    def regularization_penalty(self, transformed_y, pred_aug):
        """Unweighted ``L_reg / lambda`` for the configured regularization.

        ``transformed_y`` holds the raw library features ``phi_j(x)``; they
        are used as-is (no coefficients, no mask).
        """
        if pred_aug is None:
            return torch.tensor(0.0, device=self.device)
        if self.cfg.training.regularization == 'orthogonal':
            return orthoreg_penalty(pred_aug, transformed_y.to(pred_aug.device))
        if self.cfg.training.regularization == 'l2':
            return l2_penalty(pred_aug)
        return torch.tensor(0.0, device=self.device)

    def get_derivative_loss(self, transformed_y, dy):
        """Data-fit loss of the hybrid prediction plus diagnostics.

        Returns a dict with
        ``total_loss``  -- joint fit ``mean_i ||dy_i - (f_phy + f_aug)(x_i)||^2``,
        ``derivative_mse`` -- the same residual averaged element-wise,
        ``phy_loss``    -- symbolic-only residual (diagnostic, not optimized),
        ``aug_loss``    -- unweighted regularization penalty ``L_reg / lambda``,
        ``pred_dy``     -- the hybrid prediction.
        """
        pred_dy, pred_phy, pred_aug = self.predict_derivative(transformed_y)
        dy = dy.to(pred_dy.device)

        residual = dy - pred_dy
        total_loss = residual.pow(2).sum(dim=-1).mean()
        derivative_mse = residual.pow(2).mean()
        phy_loss = (dy - pred_phy).pow(2).sum(dim=-1).mean()
        aug_loss = self.regularization_penalty(transformed_y, pred_aug)

        return {
            'total_loss': total_loss,
            'derivative_mse': derivative_mse,
            'phy_loss': phy_loss,
            'aug_loss': aug_loss,
            'pred_dy': pred_dy
        }

    def compute_objective(self, transformed_y, dy, include_l1=True, include_reg=True):
        """Full training objective of Algorithm 1 and its components.

        ``total_loss_reg = total_loss + mu * ||w||_1 + lambda * aug_loss``.
        The L1 and penalty terms can be switched off (structure-fixed
        retraining, or while the network is frozen).
        """
        losses = self.get_derivative_loss(transformed_y, dy)
        l1_reg = torch.norm(self.net.model_phy.coef_, p=1) * self.l1_reg_weight
        node_reg = losses['aug_loss'] * self.node_reg_weight

        total_loss_reg = losses['total_loss']
        if include_l1:
            total_loss_reg = total_loss_reg + l1_reg
        if include_reg:
            total_loss_reg = total_loss_reg + node_reg

        losses['l1_reg'] = l1_reg
        losses['node_reg'] = node_reg
        losses['total_loss_reg'] = total_loss_reg
        return losses

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

    def fix_support(self, threshold=None):
        """Threshold the coefficients and freeze the support for a refit.

        Zeroes every coefficient below ``threshold`` (default
        ``training.symbolic_threshold``) and switches to the second stage, so
        each training step runs ``retrain_parameters``: only the active
        coefficients and the network are updated, masked coefficients get a
        zero gradient and stay exactly zero.
        """
        threshold = self.threshold if threshold is None else threshold
        self.net.model_phy.threshold_coefficients(threshold)
        self.feature_mask = self.net.model_phy.mask_.detach().cpu().numpy()
        self.first_stage = False
        self.structure_fixed = True
        print(f"Support fixed: {int(self.feature_mask.sum())} active coefficients")

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

        # During an optional initial freezing period the network keeps its
        # (small) initial output inside the prediction but is not updated,
        # and its penalty is skipped.
        is_frozen = self.freeze_neural and self.current_epoch < self.freeze_epochs

        # Joint fit + mu * ||w||_1 + L_reg (Algorithm 1)
        losses = self.compute_objective(transformed_y, dy, include_l1=True, include_reg=not is_frozen)
        total_loss_backward = losses['total_loss_reg']

        if self.cfg.model.hybrid_setting:
            self.log('train/neural_frozen', 1.0 if is_frozen else 0.0, prog_bar=True)

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
        self.log(f'{prefix}/l1_reg', losses['l1_reg'], prog_bar=False)
        self.log(f'{prefix}/node_reg', losses['node_reg'], prog_bar=False)
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
        """Retrain parameters while keeping symbolic structure fixed.

        The hybrid prediction ``f_phy + f_aug`` is refit jointly on the data;
        with the support frozen, neither the L1 term nor the network penalty
        is applied.
        """
        optimizer = self.optimizers()
        optimizer.zero_grad()

        # Unpack batch data
        y, transformed_y, dy, t = batch

        # Forward pass with fixed mask (data fit only)
        losses = self.compute_objective(transformed_y, dy, include_l1=False, include_reg=False)
        total_loss_backward = losses['total_loss_reg']

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

        # Same objective as the training step (no gradients are taken here)
        losses = self.compute_objective(transformed_y, dy)

        # Use consistent logging keys
        prefix = 'retrain' if self.structure_fixed else 'train'
        self.log(f'val/{prefix}_total_loss', losses['total_loss'])
        self.log(f'val/{prefix}_phy_loss', losses['phy_loss'])
        self.log(f'val/{prefix}_aug_loss', losses['aug_loss'])
        self.log(f'val/{prefix}_l1_reg', losses['l1_reg'])
        self.log(f'val/{prefix}_total_loss_reg', losses['total_loss_reg']) # Log total loss with reg

        return losses['total_loss_reg'] # Return loss used for scheduler

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

        # Structure metrics over the in-library support (plus full support)
        support = support_metrics(pred, true, self.threshold, self.n_out_of_library_terms)

        # Only print for first batch of first test loader
        if batch_idx == 0 and dataloader_idx == 0:
            # Get significant terms
            sig_terms = [(i, feature) for i, feature in enumerate(feature_names)
                        if (np.abs(true[i]) > 0).any() or
                           (np.abs(pred[i]) > self.threshold).any()]

            print("\n=== Symbolic Regression Quality ===")
            print(f"L2: {coef_l2:.3f} (norm: {coef_l2_norm:.3f})")
            print(f"Structure: F1={support['f1']:.3f} (P={support['precision']:.3f}, "
                  f"R={support['recall']:.3f}, FPR={support['fpr']:.3f})")
            print(f"Full support (K={self.n_out_of_library_terms}): "
                  f"F1={support['full_f1']:.3f}, R={support['full_recall']:.3f}")

            if sig_terms:
                print("\nCoefficients:")
                for idx, feature in sig_terms:
                    print(f"{feature:<10}: {true[idx]} -> {pred[idx]}")

        metrics = {
            'coef_l2_dist': coef_l2,
            'coef_l2_norm': coef_l2_norm,
            'f1': support['f1'],
            'precision': support['precision'],
            'recall': support['recall'],
            'fpr': support['fpr'],
            'full_recall': support['full_recall'],
            'full_f1': support['full_f1'],
        }

        # Wandb log metrics
        for name, value in metrics.items():
            self.log(f'test/{name}', value, prog_bar=False, on_epoch=True, on_step=False)

        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Enhanced test step with better numerical stability."""
        y, transformed_y, dy, t = batch
        
        # Name of the split behind this dataloader (same order as
        # HybridDataModule.test_dataloader).
        test_set_key = self.trainer.datamodule.test_set_names[dataloader_idx]
        current_test_dataset = self.trainer.datamodule.data_dict.get(test_set_key, None)
        test_set = test_set_key.replace('_test_data', '').replace('_data', '')
        
        # Compute derivative space losses (same objective as training)
        losses = self.compute_objective(transformed_y, dy)
        l1_reg = losses['l1_reg']
        total_loss = losses['total_loss_reg']
        
        # Compute normalized derivative MSE (scale-invariant)
        dy_norm = torch.norm(dy, dim=-1).mean()
        normalized_derivative_mse = losses['derivative_mse'] / (dy_norm ** 2 + 1e-8)
        
        # Compute state space trajectory and MSE
        with torch.no_grad():
            # Get feature library from datamodule
            feature_library = self.trainer.datamodule.feature_library
            
            def dynamics(t, x):
                # Transform state to feature space
                x_features = feature_library.transform(x.cpu().numpy())
                x_features = torch.tensor(x_features, dtype=torch.float32, device=x.device)
                
                # Hybrid prediction f_phy + f_aug
                pred_dy, _, _ = self.predict_derivative(x_features)
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
            print(f"Derivative Space MSE: {losses['derivative_mse']:.3e}")
            print(f"Normalized Derivative MSE: {normalized_derivative_mse:.3e}")
            print(f"State Space MSE: {state_mse:.3e}")
            print(f"Normalized State MSE: {normalized_state_mse:.3e}")
            
            # Log both derivative and state space metrics
            wandb = get_wandb()
            if wandb is not None and wandb.run is not None:
                metrics = {
                    f'test/{test_set}/derivative_mse/dataloader_idx_{dataloader_idx}': losses['derivative_mse'],
                    f'test/{test_set}/normalized_derivative_mse/dataloader_idx_{dataloader_idx}': normalized_derivative_mse,
                    f'test/{test_set}/state_mse/dataloader_idx_{dataloader_idx}': state_mse,
                    f'test/{test_set}/normalized_state_mse/dataloader_idx_{dataloader_idx}': normalized_state_mse,
                    f'test/{test_set}/phy_loss/dataloader_idx_{dataloader_idx}': losses['phy_loss'],
                    f'test/{test_set}/aug_loss/dataloader_idx_{dataloader_idx}': losses['aug_loss'],
                    f'test/{test_set}/l1_reg/dataloader_idx_{dataloader_idx}': l1_reg
                }
                wandb.log(metrics)
            
            # Also log to Lightning for terminal output
            self.log(f'test/{test_set}/derivative_mse', losses['derivative_mse'], prog_bar=False, on_epoch=True, on_step=False)
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
            wandb = get_wandb()
            if wandb is not None and wandb.run is not None:
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
