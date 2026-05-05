"""
Unit tests for orthogonal projection - the core contribution of OrthoReg.

These tests verify that the orthogonal regularization correctly enforces
orthogonality between the neural augmentation and symbolic basis functions.
"""

import pytest
import numpy as np
import torch
import pysindy as ps
from orthoreg.regularization.projection import (
    compute_empirical_inner_product,
    compute_feature_gram_matrix,
    project_onto_feature_space,
    orthogonalize_function
)


class TestEmpiricalInnerProduct:
    """Test empirical inner product computation."""
    
    def test_inner_product_with_self(self):
        """<f, f> equals the empirical L2 norm squared."""
        f = np.random.randn(10, 20)
        inner_prod = compute_empirical_inner_product(f, f)
        expected = np.mean(f * f)
        np.testing.assert_allclose(inner_prod, expected)

    def test_inner_product_symmetry(self):
        """<f, g> = <g, f>."""
        f = np.random.randn(10, 20)
        g = np.random.randn(10, 20)
        ip_fg = compute_empirical_inner_product(f, g)
        ip_gf = compute_empirical_inner_product(g, f)
        np.testing.assert_allclose(ip_fg, ip_gf)

    def test_inner_product_linearity(self):
        """<alpha * f, g> = alpha * <f, g>."""
        f = np.random.randn(10, 20)
        g = np.random.randn(10, 20)
        alpha = 2.5
        
        ip_fg = compute_empirical_inner_product(f, g)
        ip_alpha_fg = compute_empirical_inner_product(alpha * f, g)
        
        np.testing.assert_allclose(ip_alpha_fg, alpha * ip_fg)
    
    def test_orthogonal_vectors_zero_inner_product(self):
        """Test that orthogonal vectors have zero inner product."""
        # Create simple orthogonal vectors
        t = np.linspace(0, 2*np.pi, 100)
        f = np.sin(t).reshape(1, -1)
        g = np.cos(t).reshape(1, -1)
        
        # sin and cos are orthogonal over a full period
        ip = compute_empirical_inner_product(f, g)
        np.testing.assert_allclose(ip, 0.0, atol=1e-10)


class TestFeatureGramMatrix:
    """Test Gram matrix computation for feature libraries."""
    
    def test_gram_matrix_positive_definite(self):
        """Test that Gram matrix is positive semi-definite."""
        # Create simple polynomial library
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        # Generate simple trajectories
        t = np.linspace(0, 1, 50)
        x = np.sin(2*np.pi*t).reshape(1, -1, 1)
        
        G, features = compute_feature_gram_matrix(feature_lib, x, t)
        
        # Check that all eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues >= -1e-10), "Gram matrix should be positive semi-definite"
    
    def test_gram_matrix_symmetric(self):
        """Test that Gram matrix is symmetric."""
        feature_lib = ps.PolynomialLibrary(degree=3, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.random.randn(2, 50, 2)
        
        G, features = compute_feature_gram_matrix(feature_lib, x, t)
        
        np.testing.assert_allclose(G, G.T, rtol=1e-10)
    
    def test_gram_matrix_shape(self):
        """Test that Gram matrix has correct shape."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.random.randn(3, 50, 2)
        
        G, features = compute_feature_gram_matrix(feature_lib, x, t)
        
        n_features = len(features)
        assert G.shape == (n_features, n_features)


class TestOrthogonalProjection:
    """Test orthogonal projection onto feature space."""
    
    def test_projection_idempotent(self):
        """Test that projecting twice gives the same result as projecting once."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.sin(2*np.pi*t).reshape(1, -1, 1)
        
        # Create a function to project
        g = np.random.randn(1, 50)
        
        # Project once
        proj1, res1 = project_onto_feature_space(g, feature_lib, x, t)
        
        # Project the projection
        proj2, res2 = project_onto_feature_space(proj1, feature_lib, x, t)
        
        # Should get the same result
        np.testing.assert_allclose(proj1, proj2, rtol=1e-8)
    
    def test_projection_plus_residual_equals_original(self):
        """Test that projection + residual = original function."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.random.randn(2, 50, 1)
        g = np.random.randn(2, 50)
        
        proj, residual = project_onto_feature_space(g, feature_lib, x, t)
        
        reconstruction = proj + residual
        np.testing.assert_allclose(reconstruction, g, rtol=1e-8)
    
    def test_residual_orthogonal_to_features(self):
        """Test that residual is orthogonal to all feature basis functions.
        
        This is the CORE property that OrthoReg guarantees!
        """
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 100)
        x = np.sin(2*np.pi*t).reshape(1, -1, 1)
        g = np.random.randn(1, 100)
        
        proj, residual = project_onto_feature_space(g, feature_lib, x, t)
        
        # Get feature values
        feature_lib.fit(x, t)
        features = feature_lib.transform(x)
        
        # Check orthogonality with each feature
        n_features = features.shape[-1]
        for i in range(n_features):
            feature_vals = features[:, :, i]
            inner_prod = compute_empirical_inner_product(
                residual.flatten(),
                feature_vals.flatten()
            )
            np.testing.assert_allclose(
                inner_prod, 0.0, atol=1e-8,
                err_msg=f"Residual should be orthogonal to feature {i}"
            )
    
    def test_projection_in_span(self):
        """Test that projection lies in the span of the features."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.random.randn(1, 50, 1)
        
        # Create a function that's already in the span (linear combination of features)
        feature_lib.fit(x, t)
        features = feature_lib.transform(x)
        
        # Random linear combination
        coeffs = np.random.randn(features.shape[-1])
        g = np.sum(features[0, :, :] * coeffs, axis=1, keepdims=False).reshape(1, -1)
        
        proj, residual = project_onto_feature_space(g, feature_lib, x, t)
        
        # Projection should equal original, residual should be zero
        np.testing.assert_allclose(proj, g, rtol=1e-6)
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


class TestOrthogonalize:
    """Test the orthogonalization function."""
    
    def test_orthogonalize_output_orthogonal(self):
        """Test that orthogonalized function is orthogonal to feature space."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 100)
        x = np.random.randn(2, 100, 1)
        g = np.random.randn(2, 100)
        
        g_orth = orthogonalize_function(g, feature_lib, x, t)
        
        # Check orthogonality
        feature_lib.fit(x, t)
        features = feature_lib.transform(x)
        
        n_features = features.shape[-1]
        for i in range(n_features):
            feature_vals = features[:, :, i]
            inner_prod = compute_empirical_inner_product(
                g_orth.flatten(),
                feature_vals.flatten()
            )
            np.testing.assert_allclose(
                inner_prod, 0.0, atol=1e-8,
                err_msg=f"Orthogonalized function should be orthogonal to feature {i}"
            )
    
    def test_orthogonalize_preserves_orthogonal_component(self):
        """Test that orthogonalizing an already orthogonal function doesn't change it."""
        feature_lib = ps.PolynomialLibrary(degree=1, include_bias=False)
        
        t = np.linspace(0, 2*np.pi, 100)
        x = np.sin(t).reshape(1, -1, 1)
        
        # Create a function orthogonal to linear features (e.g., quadratic)
        g = (np.sin(t)**2 - 0.5).reshape(1, -1)  # Zero mean quadratic
        
        g_orth = orthogonalize_function(g, feature_lib, x, t)
        
        # Should be very similar since g was already mostly orthogonal
        correlation = np.corrcoef(g.flatten(), g_orth.flatten())[0, 1]
        assert correlation > 0.9, "Orthogonal function should not change much"


class TestNumericalStability:
    """Test numerical stability of orthogonal projection."""
    
    def test_nearly_collinear_features(self):
        """Test behavior with nearly collinear features."""
        # This tests the pseudo-inverse fallback
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=True)
        
        # Create trajectory where features are nearly collinear
        t = np.linspace(0, 0.1, 50)  # Small range
        x = (t + 1).reshape(1, -1, 1)  # Near constant
        
        g = np.random.randn(1, 50)
        
        # Should not crash, even if Gram matrix is ill-conditioned
        proj, residual = project_onto_feature_space(g, feature_lib, x, t)
        
        # Basic sanity check
        assert proj.shape == g.shape
        assert residual.shape == g.shape
    
    def test_zero_function(self):
        """Test projection of zero function."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 50)
        x = np.random.randn(1, 50, 1)
        g = np.zeros((1, 50))
        
        proj, residual = project_onto_feature_space(g, feature_lib, x, t)
        
        np.testing.assert_allclose(proj, 0.0, atol=1e-10)
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


class TestOrthogonalityVsL2:
    """Tests demonstrating the difference between orthogonal and L2 regularization.
    
    These tests show WHY orthogonal regularization is necessary in non-convex settings.
    """
    
    def test_small_l2_norm_not_orthogonal(self):
        """Demonstrate that small L2 norm doesn't guarantee orthogonality.
        
        This is the key insight: minimizing ||f_aug|| does not ensure f_aug is perpendicular to F_phy.
        """
        feature_lib = ps.PolynomialLibrary(degree=1, include_bias=False)
        
        t = np.linspace(0, 1, 100)
        x = np.linspace(-1, 1, 100).reshape(1, -1, 1)
        
        # Create a function with small norm but NOT orthogonal to linear features
        # f(x) = 0.01 * x  (small but parallel to feature space)
        f_small = 0.01 * x.flatten().reshape(1, -1)
        
        # Check it has small L2 norm
        l2_norm = np.sqrt(np.mean(f_small ** 2))
        assert l2_norm < 0.02, "Function has small L2 norm"
        
        # But check it's NOT orthogonal to features
        feature_lib.fit(x, t)
        features = feature_lib.transform(x)
        
        # Should have non-zero inner product with linear feature
        inner_prod = compute_empirical_inner_product(
            f_small.flatten(),
            features[0, :, 0].flatten()
        )
        assert abs(inner_prod) > 1e-6, "Small L2 norm doesn't guarantee orthogonality"
    
    def test_orthogonal_enforces_zero_overlap(self):
        """Show that orthogonal projection enforces zero overlap."""
        feature_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
        
        t = np.linspace(0, 1, 100)
        x = np.random.randn(1, 100, 2)
        
        # Start with ANY function (even one with large overlap)
        g = np.random.randn(1, 100) * 10  # Large magnitude
        
        # Orthogonalize it
        g_orth = orthogonalize_function(g, feature_lib, x, t)
        
        # Check that orthogonalized version has ZERO overlap
        feature_lib.fit(x, t)
        features = feature_lib.transform(x)
        
        max_inner_prod = 0
        for i in range(features.shape[-1]):
            inner_prod = abs(compute_empirical_inner_product(
                g_orth.flatten(),
                features[0, :, i].flatten()
            ))
            max_inner_prod = max(max_inner_prod, inner_prod)
        
        assert max_inner_prod < 1e-8, "Orthogonal projection enforces zero overlap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

