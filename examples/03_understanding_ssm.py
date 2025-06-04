#!/usr/bin/env python3
"""
Minimal SSM Implementation Example

Shows the core State Space Model concepts and selective scan operation
used in Mamba2MacOS. Educational example for understanding SSM fundamentals.
"""

import torch
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def minimal_ssm_example():
    """Demonstrate minimal SSM computation"""
    print("🧠 Minimal State Space Model Example")
    print("=" * 45)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # SSM dimensions
    batch_size, seq_len, d_model, d_state = 2, 16, 64, 32
    
    # Input sequence
    x = torch.randn(batch_size, d_model, seq_len, device=device)
    
    # SSM parameters
    A = -torch.exp(torch.randn(d_model, d_state, device=device))  # Stability matrix
    B = torch.randn(batch_size, 1, d_state, seq_len, device=device)  # Input matrix
    C = torch.randn(batch_size, 1, d_state, seq_len, device=device)  # Output matrix
    D = torch.ones(d_model, device=device)  # Feedthrough
    
    # Delta (timestep) - controls how much state updates
    dt = F.softplus(torch.randn(batch_size, d_model, seq_len, device=device))
    
    print(f"Input shape (x):     {x.shape}")
    print(f"State matrix (A):    {A.shape}")
    print(f"Input matrix (B):    {B.shape}")
    print(f"Output matrix (C):   {C.shape}")
    print(f"Feedthrough (D):     {D.shape}")
    print(f"Timestep (dt):       {dt.shape}")
    
    # Selective scan operation
    y = selective_scan_fn(
        x, dt, A, B, C, D,
        z=None,  # No gating
        delta_bias=None,
        delta_softplus=False  # Already applied softplus
    )
    
    print(f"\nOutput shape (y):    {y.shape}")
    print(f"✅ Selective scan successful!")
    
    return y

def demonstrate_gating():
    """Show gating mechanism (z parameter)"""
    print(f"\n🚪 Gating Mechanism Demo")
    print("=" * 30)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size, seq_len, d_model, d_state = 2, 16, 64, 32
    
    x = torch.randn(batch_size, d_model, seq_len, device=device)
    z = torch.sigmoid(torch.randn(batch_size, d_model, seq_len, device=device))  # Gate values
    
    A = -torch.exp(torch.randn(d_model, d_state, device=device))
    B = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    C = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    D = torch.ones(d_model, device=device)
    dt = F.softplus(torch.randn(batch_size, d_model, seq_len, device=device))
    
    # Without gating
    y_no_gate = selective_scan_fn(x, dt, A, B, C, D, z=None)
    
    # With gating
    y_gated = selective_scan_fn(x, dt, A, B, C, D, z=z)
    
    print(f"Without gating: {y_no_gate.shape}")
    print(f"With gating:    {y_gated.shape}")
    print(f"Gate values range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"✅ Gating demonstration complete!")

def main():
    print("📚 Understanding Mamba2MacOS SSM Components")
    print("=" * 50)
    
    # Basic SSM
    minimal_ssm_example()
    
    # Gating mechanism
    demonstrate_gating()
    
    print(f"\n🎯 Key Takeaways:")
    print(f"  • SSM processes sequences through state evolution")
    print(f"  • A matrix controls state stability (eigenvalues < 0)")
    print(f"  • B, C matrices control input/output transformations")
    print(f"  • dt (delta) controls state update rate")
    print(f"  • Gating (z) provides selective information flow")
    print(f"  • Mamba2MacOS combines these for efficient sequence modeling")

if __name__ == "__main__":
    main() 