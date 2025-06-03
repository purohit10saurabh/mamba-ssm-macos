#!/usr/bin/env python3
"""
Advanced Selective Scan Analysis for Mamba2MacOS

Deep dive into the selective scan operation that powers Mamba2.
Demonstrates different aspects of the selective scan mechanism
and how parameters affect the computation.
"""

import numpy as np
import torch
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def analyze_parameter_effects():
    """Analyze how different parameters affect selective scan output"""
    print("🔬 Selective Scan Parameter Analysis")
    print("=" * 50)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size, seq_len, d_model, d_state = 1, 32, 16, 8
    
    # Fixed input
    x = torch.randn(batch_size, d_model, seq_len, device=device)
    
    # Base parameters
    A_base = -torch.exp(torch.randn(d_model, d_state, device=device))
    B_base = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    C_base = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    D_base = torch.ones(d_model, device=device)
    dt_base = F.softplus(torch.randn(batch_size, d_model, seq_len, device=device))
    
    # Baseline computation
    y_base = selective_scan_fn(x, dt_base, A_base, B_base, C_base, D_base)
    
    print(f"Baseline output norm: {y_base.norm().item():.4f}")
    
    # Effect of A matrix scaling
    print(f"\n📊 Effect of A matrix scaling:")
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        A_scaled = A_base * scale
        y_scaled = selective_scan_fn(x, dt_base, A_scaled, B_base, C_base, D_base)
        print(f"  A scale {scale:3.1f}: output norm {y_scaled.norm().item():.4f}")
    
    # Effect of dt scaling
    print(f"\n⏱️  Effect of timestep (dt) scaling:")
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        dt_scaled = dt_base * scale
        y_dt = selective_scan_fn(x, dt_scaled, A_base, B_base, C_base, D_base)
        print(f"  dt scale {scale:3.1f}: output norm {y_dt.norm().item():.4f}")
    
    # Effect of B matrix scaling
    print(f"\n📥 Effect of B matrix scaling:")
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        B_scaled = B_base * scale
        y_B = selective_scan_fn(x, dt_base, A_base, B_scaled, C_base, D_base)
        print(f"  B scale {scale:3.1f}: output norm {y_B.norm().item():.4f}")

def demonstrate_state_evolution():
    """Show how internal state evolves during selective scan"""
    print(f"\n🔄 State Evolution During Selective Scan")
    print("=" * 45)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size, seq_len, d_model, d_state = 1, 8, 4, 2
    
    # Simple input pattern
    x = torch.zeros(batch_size, d_model, seq_len, device=device)
    x[0, 0, :4] = 1.0  # Impulse in first half
    
    # Parameters
    A = -torch.ones(d_model, d_state, device=device) * 0.5  # Stable decay
    B = torch.ones(batch_size, 1, d_state, seq_len, device=device) * 0.5
    C = torch.ones(batch_size, 1, d_state, seq_len, device=device) * 0.5
    D = torch.zeros(d_model, device=device)  # No feedthrough
    dt = torch.ones(batch_size, d_model, seq_len, device=device) * 0.1
    
    y = selective_scan_fn(x, dt, A, B, C, D)
    
    print(f"Input sequence (channel 0):")
    print(f"  {x[0, 0, :].cpu().numpy()}")
    print(f"Output sequence (channel 0):")
    print(f"  {y[0, 0, :].cpu().numpy()}")
    print(f"Notice how output decays after input impulse ends")

def compare_gating_strategies():
    """Compare different gating strategies"""
    print(f"\n🚪 Gating Strategy Comparison")
    print("=" * 35)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size, seq_len, d_model, d_state = 1, 16, 8, 4
    
    x = torch.randn(batch_size, d_model, seq_len, device=device)
    A = -torch.exp(torch.randn(d_model, d_state, device=device))
    B = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    C = torch.randn(batch_size, 1, d_state, seq_len, device=device)
    D = torch.ones(d_model, device=device) * 0.1
    dt = F.softplus(torch.randn(batch_size, d_model, seq_len, device=device))
    
    # No gating
    y_no_gate = selective_scan_fn(x, dt, A, B, C, D, z=None)
    
    # Uniform gating
    z_uniform = torch.ones_like(x) * 0.5
    y_uniform = selective_scan_fn(x, dt, A, B, C, D, z=z_uniform)
    
    # Random gating
    z_random = torch.sigmoid(torch.randn_like(x))
    y_random = selective_scan_fn(x, dt, A, B, C, D, z=z_random)
    
    # Learned pattern gating
    z_pattern = torch.sigmoid(torch.sin(torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(0) * 0.5))
    z_pattern = z_pattern.expand_as(x)
    y_pattern = selective_scan_fn(x, dt, A, B, C, D, z=z_pattern)
    
    print(f"Output norms:")
    print(f"  No gating:      {y_no_gate.norm().item():.4f}")
    print(f"  Uniform (0.5):  {y_uniform.norm().item():.4f}")
    print(f"  Random gating:  {y_random.norm().item():.4f}")
    print(f"  Pattern gating: {y_pattern.norm().item():.4f}")

def analyze_computational_complexity():
    """Analyze computational complexity for different sequence lengths"""
    print(f"\n⚡ Computational Complexity Analysis")
    print("=" * 40)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_model, d_state = 64, 32
    batch_size = 1
    
    import time
    
    seq_lengths = [16, 32, 64, 128, 256, 512]
    times = []
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, d_model, seq_len, device=device)
        A = -torch.exp(torch.randn(d_model, d_state, device=device))
        B = torch.randn(batch_size, 1, d_state, seq_len, device=device)
        C = torch.randn(batch_size, 1, d_state, seq_len, device=device)
        D = torch.ones(d_model, device=device)
        dt = F.softplus(torch.randn(batch_size, d_model, seq_len, device=device))
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = selective_scan_fn(x, dt, A, B, C, D)
        
        if device == "mps":
            torch.mps.synchronize()
        
        start_time = time.time()
        num_runs = 20
        for _ in range(num_runs):
            with torch.no_grad():
                _ = selective_scan_fn(x, dt, A, B, C, D)
        
        if device == "mps":
            torch.mps.synchronize()
        
        avg_time = (time.time() - start_time) / num_runs
        times.append(avg_time)
        
        print(f"  seq_len {seq_len:3d}: {avg_time*1000:.2f}ms")
    
    # Analyze scaling
    print(f"\n📈 Scaling Analysis:")
    for i in range(1, len(seq_lengths)):
        ratio = times[i] / times[i-1]
        length_ratio = seq_lengths[i] / seq_lengths[i-1]
        print(f"  {seq_lengths[i-1]} → {seq_lengths[i]}: {ratio:.2f}x time ({length_ratio:.1f}x length)")

def main():
    print("🧬 Advanced Selective Scan Analysis for Mamba2MacOS")
    print("=" * 60)
    
    # Parameter effects
    analyze_parameter_effects()
    
    # State evolution
    demonstrate_state_evolution()
    
    # Gating strategies
    compare_gating_strategies()
    
    # Computational complexity
    analyze_computational_complexity()
    
    print(f"\n🎯 Key Insights:")
    print(f"  • A matrix controls state decay/stability")
    print(f"  • dt controls update rate (smaller = more selective)")
    print(f"  • B/C matrices control input/output coupling")
    print(f"  • Gating provides content-dependent selectivity")
    print(f"  • Computational complexity scales roughly linearly")
    print(f"  • Selective scan enables efficient long sequences")

if __name__ == "__main__":
    main() 