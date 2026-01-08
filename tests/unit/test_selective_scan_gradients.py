"""
Unit tests for selective scan gradient flow verification.
These tests ensure that gradients computed via different methods
are consistent and correctly propagated through the selective scan operation.
"""

import unittest

import torch

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class TestSelectiveScanGradients(unittest.TestCase):
    """Test gradient flow through selective scan operations."""

    def test_gradient_matches_explicit_computation(self):
        """Verify backward() and torch.autograd.grad() produce identical gradients."""
        batch, dim, dstate, seq_len = 2, 4, 8, 6
        torch.manual_seed(42)

        u_base = torch.randn(batch, dim, seq_len)
        delta_base = torch.randn(batch, dim, seq_len)
        A_base = torch.randn(dim, dstate)
        B_base = torch.randn(dim, dstate)
        C_base = torch.randn(dim, dstate)

        # Method 1: loss.backward()
        u1 = u_base.clone().requires_grad_(True)
        delta1 = delta_base.clone().requires_grad_(True)
        A1 = A_base.clone().requires_grad_(True)
        B1 = B_base.clone().requires_grad_(True)
        C1 = C_base.clone().requires_grad_(True)

        out1 = selective_scan_fn(u1, delta1, A1, B1, C1)
        loss1 = out1.sum()
        loss1.backward()

        grad_u1 = u1.grad.clone()
        grad_delta1 = delta1.grad.clone()
        grad_A1 = A1.grad.clone()
        grad_B1 = B1.grad.clone()
        grad_C1 = C1.grad.clone()

        # Method 2: torch.autograd.grad()
        u2 = u_base.clone().requires_grad_(True)
        delta2 = delta_base.clone().requires_grad_(True)
        A2 = A_base.clone().requires_grad_(True)
        B2 = B_base.clone().requires_grad_(True)
        C2 = C_base.clone().requires_grad_(True)

        out2 = selective_scan_fn(u2, delta2, A2, B2, C2)
        loss2 = out2.sum()

        (grad_u2, grad_delta2, grad_A2, grad_B2, grad_C2) = torch.autograd.grad(
            loss2, [u2, delta2, A2, B2, C2], retain_graph=False
        )

        # Verify gradients match
        torch.testing.assert_close(grad_u1, grad_u2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad_delta1, grad_delta2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad_A1, grad_A2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad_B1, grad_B2, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(grad_C1, grad_C2, atol=1e-6, rtol=1e-6)

        # Verify gradients are non-zero
        self.assertGreater(grad_u1.abs().mean().item(), 1e-7)
        self.assertGreater(grad_delta1.abs().mean().item(), 1e-7)
        self.assertGreater(grad_A1.abs().mean().item(), 1e-7)
        self.assertGreater(grad_B1.abs().mean().item(), 1e-7)
        self.assertGreater(grad_C1.abs().mean().item(), 1e-7)


def run_selective_scan_gradient_tests():
    """Run all gradient tests with verbose output."""
    print("🧪 Running selective scan gradient tests...")

    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSelectiveScanGradients)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("✅ All selective scan gradient tests passed")
            return True
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return False


if __name__ == "__main__":
    run_selective_scan_gradient_tests()
