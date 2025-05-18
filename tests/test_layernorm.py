import unittest

import torch

import mamba_ssm


class TestLayerNorm(unittest.TestCase):
    """Tests specifically for the layernorm_gated module."""

    def test_rms_norm_fallback(self):
        """Test the RMSNorm fallback implementation."""
        from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

        # Test basic case
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        x = torch.randn(batch, seq_len, hidden_size)
        weight = torch.ones(hidden_size)
        bias = torch.zeros(hidden_size)
        
        out = rms_norm_ref(x, weight, bias)
        
        # Check output shape
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Test with different weight values
        weight = torch.randn(hidden_size).abs()  # Random positive weights
        out = rms_norm_ref(x, weight, bias)
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Test with gating
        z = torch.randn(batch, seq_len, hidden_size)
        
        # Test with norm_before_gate=True
        out1 = rms_norm_ref(x, weight, bias, z, norm_before_gate=True)
        self.assertEqual(out1.shape, (batch, seq_len, hidden_size))
        
        # Test with norm_before_gate=False
        out2 = rms_norm_ref(x, weight, bias, z, norm_before_gate=False)
        self.assertEqual(out2.shape, (batch, seq_len, hidden_size))
        
        # They should be different
        self.assertTrue(torch.any(torch.ne(out1, out2)))
    
    def test_rmsnorm_fn(self):
        """Test the rmsnorm_fn function that should use the fallback on macOS."""
        from mamba_ssm.ops.triton.layernorm_gated import (rms_norm_ref,
                                                          rmsnorm_fn)
        
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        x = torch.randn(batch, seq_len, hidden_size)
        weight = torch.ones(hidden_size)
        bias = None  # RMSNorm typically doesn't use bias
        
        # Run both implementations
        out1 = rmsnorm_fn(x, weight, bias)
        out2 = rms_norm_ref(x, weight, bias)
        
        # They should be close
        self.assertTrue(torch.allclose(out1, out2, rtol=1e-5, atol=1e-5))
    
    def test_layernorm_fn(self):
        """Test the layernorm_fn function that should use the fallback on macOS."""
        from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn
        
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        x = torch.randn(batch, seq_len, hidden_size)
        weight = torch.ones(hidden_size)
        bias = torch.zeros(hidden_size)
        
        # Run layernorm_fn
        out = layernorm_fn(x, weight, bias)
        
        # Check output shape
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Compare with PyTorch's layernorm
        torch_ln = torch.nn.LayerNorm(hidden_size)
        torch_ln.weight.data = weight.clone()
        torch_ln.bias.data = bias.clone()
        expected = torch_ln(x)
        
        # They should be close - use a more relaxed tolerance for macOS fallback implementations
        self.assertTrue(torch.allclose(out, expected, rtol=1e-4, atol=1e-4))
    
    def test_rmsnorm_module(self):
        """Test the RMSNorm module."""
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
        
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        # Create module
        rms_norm = RMSNorm(hidden_size)
        
        # Test forward
        x = torch.randn(batch, seq_len, hidden_size)
        out = rms_norm(x)
        
        # Check output shape
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Test with gating
        z = torch.randn(batch, seq_len, hidden_size)
        out = rms_norm(x, z)
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
    
    def test_layernorm_module(self):
        """Test the LayerNorm module."""
        from mamba_ssm.ops.triton.layernorm_gated import LayerNorm
        
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        # Create module
        layer_norm = LayerNorm(hidden_size)
        
        # Test forward
        x = torch.randn(batch, seq_len, hidden_size)
        out = layer_norm(x)
        
        # Check output shape
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Test with gating
        z = torch.randn(batch, seq_len, hidden_size)
        out = layer_norm(x, z)
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))


if __name__ == "__main__":
    unittest.main() 