#!/usr/bin/env python3
"""
Basic Core Modules Usage Example

Demonstrates fundamental usage patterns for core Mamba modules on Apple Silicon.
Shows both Mamba 1 (Mamba) and Mamba 2 (Mamba2) architectures.
"""

import torch

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba


def demo_mamba1(device):
    print("\n🔷 Mamba 1 (SSM Architecture)")
    print("=" * 40)
    block = Mamba(
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_min=0.001,
        dt_max=0.1,
        device=device,
    )
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    input_features = torch.randn(8, 1024, 512, device=device)
    with torch.no_grad():
        output = block(input_features)
    print(f"Input shape:  {input_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0, 0, :5]}")
    print("✅ Mamba 1 forward pass successful!")


def demo_mamba2(device):
    print("\n🔶 Mamba 2 (SSD Architecture)")
    print("=" * 40)
    block = Mamba2(
        d_model=512,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        chunk_size=256,
        device=device,
    )
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    input_features = torch.randn(8, 1024, 512, device=device)
    with torch.no_grad():
        output = block(input_features)
    print(f"Input shape:  {input_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0, 0, :5]}")
    print("✅ Mamba 2 forward pass successful!")
    print("\n🔄 Step function demo:")
    conv_state, ssm_state = block.allocate_inference_cache(8, 1024)
    x_step = input_features[:, :1, :]
    with torch.no_grad():
        y_step, new_conv_state, new_ssm_state = block.step(x_step, conv_state, ssm_state)
    print(f"Step input:  {x_step.shape}")
    print(f"Step output: {y_step.shape}")
    print("✅ Step function successful!")


def main():
    print("🍎 Basic Core Modules Usage")
    print("=" * 50)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    demo_mamba1(device)
    demo_mamba2(device)

    print("\n📊 Summary")
    print("=" * 50)
    print("✅ Mamba 1: SSM architecture, d_state=16")
    print("✅ Mamba 2: SSD architecture, d_state=64, supports step()")


if __name__ == "__main__":
    main()
