import unittest

import torch

import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import (mamba_inner_fn,
                                                    selective_scan_fn,
                                                    selective_scan_ref)


class TestMambaSSMBasic(unittest.TestCase):
    """Basic tests for mamba_ssm that should work on macOS Apple Silicon."""

    def test_version(self):
        """Test that the package version is available."""
        self.assertIsNotNone(mamba_ssm.__version__)
        self.assertTrue(isinstance(mamba_ssm.__version__, str))
    
    def test_selective_scan_ref(self):
        """Test that the reference implementation of selective_scan works."""
        batch_size = 2
        dim = 4
        seqlen = 8
        dstate = 3
        
        # Create some dummy inputs
        u = torch.randn(batch_size, dim, seqlen)
        delta = torch.randn(batch_size, dim, seqlen).abs()
        A = torch.randn(dim, dstate)
        B = torch.randn(dim, dstate)
        C = torch.randn(dim, dstate)
        
        # Run the reference implementation
        out = selective_scan_ref(u, delta, A, B, C)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, dim, seqlen))
    
    def test_selective_scan_fn(self):
        """Test that selective_scan_fn works (should use ref implementation on macOS)."""
        batch_size = 2
        dim = 4
        seqlen = 8
        dstate = 3
        
        # Create some dummy inputs
        u = torch.randn(batch_size, dim, seqlen)
        delta = torch.randn(batch_size, dim, seqlen).abs()
        A = torch.randn(dim, dstate)
        B = torch.randn(dim, dstate)
        C = torch.randn(dim, dstate)
        
        # Run the function
        out = selective_scan_fn(u, delta, A, B, C)
        
        # Check output shape
        self.assertEqual(out.shape, (batch_size, dim, seqlen))
    
    def test_rms_norm_ref(self):
        """Test the RMSNorm reference implementation."""
        from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
        
        batch = 2
        seq_len = 5
        hidden_size = 8
        
        x = torch.randn(batch, seq_len, hidden_size)
        weight = torch.ones(hidden_size)
        bias = torch.zeros(hidden_size)
        
        out = rms_norm_ref(x, weight, bias)
        
        # Check output shape
        self.assertEqual(out.shape, (batch, seq_len, hidden_size))
        
        # Check normalization property
        # Each row should have variance close to 1
        normalized_var = ((out**2).mean(dim=-1) - 1.0).abs().mean()
        self.assertLess(normalized_var, 1e-5)

    @unittest.skipIf(torch.cuda.is_available(), "Skip on systems with CUDA")
    def test_fallbacks_for_cpu(self):
        """Test that importing modules that have fallbacks works."""
        # Skip Mamba2 test as it requires Triton which is not available on Apple Silicon
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            self.assertTrue(True, "Successfully imported Mamba")
        except ImportError as e:
            self.fail(f"Failed to import Mamba: {e}")

    def test_small_model_forward(self):
        """Test a small model forward pass using CPU."""
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            
            d_model = 16  # Small dimension for testing
            d_state = 4
            
            # Create a small Mamba model
            model = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=2,
                expand=2
            )
            
            # Create a small input tensor
            batch_size = 2
            seq_len = 6
            x = torch.randn(batch_size, seq_len, d_model)
            
            # Run forward pass
            out = model(x)
            
            # Check output shape
            self.assertEqual(out.shape, (batch_size, seq_len, d_model))
            
        except ImportError as e:
            self.skipTest(f"Mamba model not available: {e}")
        except Exception as e:
            self.fail(f"Error during model forward pass: {e}")


if __name__ == "__main__":
    unittest.main() 