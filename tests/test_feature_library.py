"""
Unit tests for feature library transformations.

Tests that symbolic feature libraries produce correct transformations
with expected shapes and properties.
"""

import pytest
import numpy as np
import pysindy as ps


class TestPolynomialLibrary:
    """Test polynomial feature library."""
    
    def test_polynomial_degree_1(self):
        """Test linear polynomial library."""
        lib = ps.PolynomialLibrary(degree=1, include_bias=False)
        
        # 2D state space
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape: (1, 2, 2)
        t = np.array([0.0, 1.0])
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # Should have 2 features for 2D linear (x0, x1)
        assert features.shape == (1, 2, 2)
        
        # Check feature values
        np.testing.assert_allclose(features[0, 0, :], [1.0, 2.0])
        np.testing.assert_allclose(features[0, 1, :], [3.0, 4.0])
    
    def test_polynomial_degree_2(self):
        """Test quadratic polynomial library."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        # 1D state space for simplicity
        x = np.array([[[2.0]], [[3.0]]])  # shape: (2, 1, 1)
        t = np.array([0.0])
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # Should have 2 features: x, x^2
        assert features.shape[2] == 2
        
        # Check values for x=2
        np.testing.assert_allclose(features[0, 0, :], [2.0, 4.0])
        # Check values for x=3
        np.testing.assert_allclose(features[1, 0, :], [3.0, 9.0])
    
    def test_feature_names(self):
        """Test that feature names are correctly generated."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        x = np.random.randn(1, 10, 2)
        t = np.linspace(0, 1, 10)
        
        lib.fit(x, t)
        names = lib.get_feature_names()
        
        # For 2D state with degree 2: x0, x1, x0^2, x0*x1, x1^2
        expected_count = 5
        assert len(names) == expected_count


class TestFourierLibrary:
    """Test Fourier feature library."""
    
    def test_fourier_features(self):
        """Test Fourier library creates sin/cos features."""
        lib = ps.FourierLibrary(n_frequencies=1, include_sin=True, include_cos=True)
        
        x = np.array([[[0.0], [np.pi/2], [np.pi]]])  # shape: (1, 3, 1)
        t = np.array([0.0, 1.0, 2.0])
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # Should have sin(x) and cos(x) for 1 frequency
        assert features.shape[2] >= 2
    
    def test_multiple_frequencies(self):
        """Test Fourier library with multiple frequencies."""
        lib = ps.FourierLibrary(n_frequencies=2, include_sin=True, include_cos=True)
        
        x = np.random.randn(1, 10, 1)
        t = np.linspace(0, 1, 10)
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # 2 frequencies * 2 (sin+cos) per state dimension
        # Actual count depends on library implementation
        assert features.shape[2] >= 4


class TestCustomLibrary:
    """Test custom library functionality."""
    
    def test_custom_functions(self):
        """Test library with custom functions."""
        # Create library with custom functions
        lib = ps.CustomLibrary(
            library_functions=[
                lambda x: x,
                lambda x: x**2,
                lambda x: np.sin(x)
            ],
            function_names=[
                lambda x: x,
                lambda x: f"{x}^2", 
                lambda x: f"sin({x})"
            ]
        )
        
        x = np.array([[[1.0]], [[2.0]]])  # shape: (2, 1, 1)
        t = np.array([0.0])
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # Should have 3 features
        assert features.shape[2] == 3
        
        # Check values for x=1
        expected_1 = [1.0, 1.0, np.sin(1.0)]
        np.testing.assert_allclose(features[0, 0, :], expected_1, rtol=1e-6)


class TestLibraryComposition:
    """Test composition of multiple libraries."""
    
    def test_combined_libraries(self):
        """Test combining polynomial and Fourier libraries."""
        poly_lib = ps.PolynomialLibrary(degree=1, include_bias=False)
        fourier_lib = ps.FourierLibrary(n_frequencies=1)
        
        # Combine libraries
        combined_lib = poly_lib + fourier_lib
        
        x = np.random.randn(1, 10, 1)
        t = np.linspace(0, 1, 10)
        
        combined_lib.fit(x, t)
        features = combined_lib.transform(x)
        
        # Should have features from both libraries
        assert features.shape[2] > 1


class TestFeatureLibraryProperties:
    """Test mathematical properties of feature libraries."""
    
    def test_feature_library_deterministic(self):
        """Test that feature library gives same output for same input."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        x = np.random.randn(1, 10, 2)
        t = np.linspace(0, 1, 10)
        
        lib.fit(x, t)
        features1 = lib.transform(x)
        features2 = lib.transform(x)
        
        np.testing.assert_allclose(features1, features2)
    
    def test_feature_library_preserves_batch_dim(self):
        """Test that batch dimension is preserved."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        n_trajectories = 5
        n_timepoints = 20
        n_states = 3
        
        x = np.random.randn(n_trajectories, n_timepoints, n_states)
        t = np.linspace(0, 1, n_timepoints)
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # Should preserve trajectory and time dimensions
        assert features.shape[0] == n_trajectories
        assert features.shape[1] == n_timepoints
    
    def test_feature_library_with_zeros(self):
        """Test feature library behavior with zero inputs."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        x = np.zeros((1, 10, 2))
        t = np.linspace(0, 1, 10)
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        # All polynomial features of zeros should be zero
        np.testing.assert_allclose(features, 0.0)


class TestLibraryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_timepoint(self):
        """Test library with single timepoint."""
        lib = ps.PolynomialLibrary(degree=1, include_bias=False)
        
        x = np.array([[[1.0, 2.0]]])  # shape: (1, 1, 2)
        t = np.array([0.0])
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        assert features.shape == (1, 1, 2)
    
    def test_single_trajectory(self):
        """Test library with single trajectory."""
        lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        x = np.random.randn(1, 50, 2)
        t = np.linspace(0, 1, 50)
        
        lib.fit(x, t)
        features = lib.transform(x)
        
        assert features.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

