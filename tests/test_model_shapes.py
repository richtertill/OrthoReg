"""
Unit tests for model shapes and forward passes.

Tests that hybrid models produce outputs with correct shapes
and that all components work together properly.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


class SimpleSymbolicModel(nn.Module):
    """Simple symbolic model for testing."""
    
    def __init__(self, n_features, n_states):
        super().__init__()
        self.coef_ = nn.Parameter(torch.randn(n_features, n_states))
    
    def forward(self, x):
        """
        Args:
            x: [batch, time, n_features]
        Returns:
            [batch, time, n_states]
        """
        return torch.einsum('btf,fs->bts', x, self.coef_)


class SimpleNeuralModel(nn.Module):
    """Simple neural model for testing."""
    
    def __init__(self, n_features, hidden_dim, n_states):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_states)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, time, n_features]
        Returns:
            [batch, time, n_states]
        """
        batch, time, features = x.shape
        x_flat = x.reshape(-1, features)
        out_flat = self.net(x_flat)
        return out_flat.reshape(batch, time, -1)


class SimpleHybridModel(nn.Module):
    """Simple hybrid model for testing."""
    
    def __init__(self, n_features, hidden_dim, n_states):
        super().__init__()
        self.symbolic = SimpleSymbolicModel(n_features, n_states)
        self.neural = SimpleNeuralModel(n_features, hidden_dim, n_states)
    
    def forward(self, x, return_components=False):
        """
        Args:
            x: [batch, time, n_features]
        Returns:
            If return_components:
                (total, symbolic, neural)
            Else:
                total: [batch, time, n_states]
        """
        sym_out = self.symbolic(x)
        neural_out = self.neural(x)
        total = sym_out + neural_out
        
        if return_components:
            return total, sym_out, neural_out
        return total


class TestModelShapes:
    """Test that models produce correct output shapes."""
    
    def test_symbolic_model_shape(self):
        """Test symbolic model output shape."""
        n_features = 5
        n_states = 2
        model = SimpleSymbolicModel(n_features, n_states)
        
        batch_size = 3
        seq_length = 10
        x = torch.randn(batch_size, seq_length, n_features)
        
        out = model(x)
        
        assert out.shape == (batch_size, seq_length, n_states)
    
    def test_neural_model_shape(self):
        """Test neural model output shape."""
        n_features = 5
        hidden_dim = 32
        n_states = 2
        model = SimpleNeuralModel(n_features, hidden_dim, n_states)
        
        batch_size = 4
        seq_length = 20
        x = torch.randn(batch_size, seq_length, n_features)
        
        out = model(x)
        
        assert out.shape == (batch_size, seq_length, n_states)
    
    def test_hybrid_model_shape(self):
        """Test hybrid model output shape."""
        n_features = 10
        hidden_dim = 64
        n_states = 3
        model = SimpleHybridModel(n_features, hidden_dim, n_states)
        
        batch_size = 2
        seq_length = 15
        x = torch.randn(batch_size, seq_length, n_features)
        
        out = model(x)
        
        assert out.shape == (batch_size, seq_length, n_states)
    
    def test_hybrid_model_components_shape(self):
        """Test that hybrid model returns correctly shaped components."""
        n_features = 8
        hidden_dim = 32
        n_states = 2
        model = SimpleHybridModel(n_features, hidden_dim, n_states)
        
        batch_size = 3
        seq_length = 12
        x = torch.randn(batch_size, seq_length, n_features)
        
        total, symbolic, neural = model(x, return_components=True)
        
        expected_shape = (batch_size, seq_length, n_states)
        assert total.shape == expected_shape
        assert symbolic.shape == expected_shape
        assert neural.shape == expected_shape


class TestModelForwardPass:
    """Test forward pass behavior."""
    
    def test_hybrid_decomposition(self):
        """Test that total = symbolic + neural."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(2, 10, 5)
        total, symbolic, neural = model(x, return_components=True)
        
        expected = symbolic + neural
        torch.testing.assert_close(total, expected)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(2, 10, 5)
        out = model(x)
        
        loss = out.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
    
    def test_batch_independence(self):
        """Test that different batch elements are processed independently."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model.eval()
        
        x1 = torch.randn(1, 10, 5)
        x2 = torch.randn(1, 10, 5)
        x_combined = torch.cat([x1, x2], dim=0)
        
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_combined = model(x_combined)
        
        torch.testing.assert_close(out_combined[0:1], out1)
        torch.testing.assert_close(out_combined[1:2], out2)


class TestDeviceConsistency:
    """Test device handling (CPU/GPU)."""
    
    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model = model.cpu()
        
        x = torch.randn(2, 10, 5).cpu()
        out = model(x)
        
        assert out.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model = model.cuda()
        
        x = torch.randn(2, 10, 5).cuda()
        out = model(x)
        
        assert out.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch_error(self):
        """Test that device mismatch raises error."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model = model.cuda()
        
        x = torch.randn(2, 10, 5).cpu()
        
        with pytest.raises(RuntimeError):
            out = model(x)


class TestDtypeConsistency:
    """Test data type handling."""
    
    def test_float32_forward(self):
        """Test forward pass with float32."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model = model.float()
        
        x = torch.randn(2, 10, 5).float()
        out = model(x)
        
        assert out.dtype == torch.float32
    
    def test_float64_forward(self):
        """Test forward pass with float64."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        model = model.double()
        
        x = torch.randn(2, 10, 5).double()
        out = model(x)
        
        assert out.dtype == torch.float64


class TestModelParameters:
    """Test parameter counts and shapes."""
    
    def test_symbolic_parameters(self):
        """Test symbolic model has expected parameters."""
        model = SimpleSymbolicModel(n_features=5, n_states=2)
        
        params = list(model.parameters())
        assert len(params) == 1  # Just the coefficient matrix
        assert params[0].shape == (5, 2)
    
    def test_hybrid_parameters(self):
        """Test hybrid model has both symbolic and neural parameters."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        # Count symbolic parameters
        symbolic_params = list(model.symbolic.parameters())
        assert len(symbolic_params) > 0
        
        # Count neural parameters
        neural_params = list(model.neural.parameters())
        assert len(neural_params) > 0
        
        # Total should be sum
        total_params = list(model.parameters())
        assert len(total_params) == len(symbolic_params) + len(neural_params)
    
    def test_parameter_requires_grad(self):
        """Test that parameters require gradients by default."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        for param in model.parameters():
            assert param.requires_grad


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_batch(self):
        """Test with batch size of 1."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(1, 10, 5)
        out = model(x)
        
        assert out.shape == (1, 10, 2)
    
    def test_single_timestep(self):
        """Test with single timestep."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(3, 1, 5)
        out = model(x)
        
        assert out.shape == (3, 1, 2)
    
    def test_large_batch(self):
        """Test with large batch size."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(100, 10, 5)
        out = model(x)
        
        assert out.shape == (100, 10, 2)


class TestOrthogonalRegularization:
    """Test orthogonal regularization computation."""
    
    def test_orthogonality_penalty_shape(self):
        """Test that orthogonality penalty is a scalar."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        x = torch.randn(3, 10, 5)
        total, symbolic, neural = model(x, return_components=True)
        
        # Compute simple orthogonality penalty: sum of inner products squared
        # This is a simplified version - actual implementation is more sophisticated
        penalty = torch.sum(symbolic * neural) ** 2
        
        assert penalty.shape == torch.Size([])  # Scalar
    
    def test_zero_neural_gives_zero_penalty(self):
        """Test that zero neural component gives zero orthogonality penalty."""
        model = SimpleHybridModel(n_features=5, hidden_dim=32, n_states=2)
        
        # Zero out neural network
        with torch.no_grad():
            for param in model.neural.parameters():
                param.zero_()
        
        x = torch.randn(3, 10, 5)
        total, symbolic, neural = model(x, return_components=True)
        
        # Neural should be zero
        torch.testing.assert_close(neural, torch.zeros_like(neural))
        
        # Orthogonality penalty should be zero
        penalty = torch.sum(symbolic * neural) ** 2
        torch.testing.assert_close(penalty, torch.tensor(0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

