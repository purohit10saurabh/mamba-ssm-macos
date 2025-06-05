#!/usr/bin/env python3
"""
Basic Mamba2MacOS Usage Example

Demonstrates fundamental usage patterns for Mamba2MacOS on Apple Silicon.
This is the simplest entry point for understanding the core functionality.
"""

import torch

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS


def main():
    print("🍎 Basic Mamba2MacOS Usage")
    print("=" * 40)
    
    # Setup device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create model
    model = Mamba2MacOS(
        d_model=256,
        d_state=16, 
        d_conv=4,
        expand=2,
        headdim=64,
        device=device
    )
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, 256, device=device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"✅ Basic forward pass successful!")
    
    # Step function demo
    print(f"\n🔄 Step function demo:")
    conv_state, ssm_state = model.allocate_inference_cache(batch_size, seq_len)
    
    # Single step
    x_step = x[:, :1, :]  # First token
    with torch.no_grad():
        y_step, new_conv_state, new_ssm_state = model.step(x_step, conv_state, ssm_state)
    
    print(f"Step input:  {x_step.shape}")
    print(f"Step output: {y_step.shape}")
    print(f"✅ Step function successful!")

if __name__ == "__main__":
    main() 