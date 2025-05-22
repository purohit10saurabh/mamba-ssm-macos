import platform
import time

import numpy as np
import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

print("Mamba SSM on macOS Apple Silicon - Benchmark")
print("--------------------------------------------")
print(f"PyTorch version: {torch.__version__}")
print(f"Running on: {platform.platform()}")

# Check if MPS is available
mps_available = torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")


def benchmark_operation(operation, *args, device="cpu", n_runs=10, warmup=3):
    """Benchmark a PyTorch operation by running it multiple times and measuring execution time"""
    # Move args to device
    args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
    
    # Warmup
    for _ in range(warmup):
        result = operation(*args)
    
    # Synchronize before timing
    if device == "mps":
        torch.mps.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(n_runs):
        result = operation(*args)
    
    # Synchronize after timing
    if device == "mps":
        torch.mps.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    return avg_time, result


class SimpleMambaModel(nn.Module):
    def __init__(self, d_model, n_layer, d_state, device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            ) for _ in range(n_layer)
        ])
        self.to(device)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Test 1: Basic selective scan operation
def run_selective_scan_benchmark(batch_size=2, seq_len=128, hidden_dim=16, d_state=4, n_runs=10):
    print("\n1. Selective Scan Benchmark (Reference Implementation)")
    print("--------------------------------------------------")
    
    devices = ["cpu"]
    if mps_available:
        devices.append("mps")
    
    for device in devices:
        # Create sample input
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        # Create some sample SSM parameters
        A = torch.randn(hidden_dim, d_state, device=device)
        B = torch.randn(batch_size, seq_len, hidden_dim, d_state, device=device)
        C = torch.randn(batch_size, seq_len, hidden_dim, d_state, device=device)
        D = torch.zeros(hidden_dim, device=device)
        
        # Move tensors to correct device
        input_tensor = input_tensor.to(device)
        A = A.to(device)
        B = B.to(device)
        C = C.to(device)
        D = D.to(device)
        
        # Create delta
        delta = torch.rand(batch_size, seq_len, hidden_dim, device=device)
        
        # Define operation
        def selective_scan_op(u, delta, A, B, C, D):
            # Rearrange inputs for selective_scan_fn
            u = u.transpose(1, 2)  # from (B, L, D) to (B, D, L)
            delta = delta.transpose(1, 2)  # from (B, L, D) to (B, D, L)
            output, _ = selective_scan_fn(
                u, delta, A, B, C, D,
                delta_bias=None, delta_softplus=True
            )
            return output
        
        # Run benchmark
        time_taken, output = benchmark_operation(
            selective_scan_op, 
            input_tensor, delta, A, B, C, D, 
            device=device, 
            n_runs=n_runs
        )
        
        print(f"  Device: {device}, Shape: {input_tensor.shape}")
        print(f"  Average time: {time_taken*1000:.3f} ms")
        print(f"  Output shape: {output.shape}")


# Test 2: Simple Mamba model benchmark
def run_mamba_model_benchmark(batch_sizes=[1, 2, 4], seq_lens=[64, 128, 256], d_model=64, n_layer=2, d_state=16, n_runs=5):
    print("\n2. Simple Mamba Model Benchmark")
    print("------------------------------")
    
    devices = ["cpu"]
    if mps_available:
        devices.append("mps")
    
    results = []
    
    for device in devices:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                # Create model
                model = SimpleMambaModel(d_model=d_model, n_layer=n_layer, d_state=d_state, device=device)
                
                # Create input
                x = torch.randn(batch_size, d_model, seq_len, device=device)
                
                # Define forward operation
                def forward_op(x):
                    with torch.no_grad():
                        return model(x)
                
                # Run benchmark
                time_taken, output = benchmark_operation(
                    forward_op, 
                    x, 
                    device=device, 
                    n_runs=n_runs
                )
                
                results.append({
                    'device': device,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'time_ms': time_taken * 1000,
                    'tokens_per_sec': (batch_size * seq_len) / time_taken
                })
                
                print(f"  Device: {device}, Batch: {batch_size}, Seq Length: {seq_len}")
                print(f"  Average time: {time_taken*1000:.3f} ms")
                print(f"  Tokens per second: {(batch_size * seq_len) / time_taken:.1f}")
    
    if mps_available:
        # Calculate speedup for MPS vs CPU
        print("\nSpeedup with MPS vs CPU:")
        for i in range(len(results) // 2):
            cpu_result = results[i]
            mps_result = results[i + len(results) // 2]
            speedup = mps_result['tokens_per_sec'] / cpu_result['tokens_per_sec']
            print(f"  Batch: {cpu_result['batch_size']}, Seq Length: {cpu_result['seq_len']}")
            print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("\nRunning benchmarks...")
    run_selective_scan_benchmark()
    run_mamba_model_benchmark() 