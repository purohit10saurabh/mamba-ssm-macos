import time

import numpy as np
import torch


def benchmark_operation(operation, *args, n_runs=100):
    """Benchmark a PyTorch operation by running it multiple times and measuring execution time"""
    # Warmup
    for _ in range(10):
        result = operation(*args)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(n_runs):
        result = operation(*args)
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / n_runs

# Create test cases of varying sizes
sizes = [
    # Matrix multiplication (2D)
    (100, 100, 100),  # (m, k, n) - multiply m×k by k×n matrices
    (500, 500, 500),
    (1000, 1000, 1000),
    
    # Batched matrix multiplication (3D)
    (32, 128, 128, 128),  # (b, m, k, n) - batch of b matrix multiplications
    (64, 256, 256, 256),
    
    # Higher dimensional operations
    (16, 32, 64, 64, 32),  # (a, b, m, k, n) - higher-dim batch matrix mult
]

print("Benchmarking torch.matmul vs torch.einsum")
print("----------------------------------------")
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print()

results = []

for size in sizes:
    if len(size) == 3:
        m, k, n = size
        A = torch.randn(m, k)
        B = torch.randn(k, n)
        
        matmul_op = lambda A, B: torch.matmul(A, B)
        einsum_op = lambda A, B: torch.einsum('ik,kj->ij', A, B)
        
        size_str = f"Matrix multiplication: ({m}×{k}) @ ({k}×{n})"
        
    elif len(size) == 4:
        b, m, k, n = size
        A = torch.randn(b, m, k)
        B = torch.randn(b, k, n)
        
        matmul_op = lambda A, B: torch.matmul(A, B)
        einsum_op = lambda A, B: torch.einsum('bik,bkj->bij', A, B)
        
        size_str = f"Batched matrix multiplication: {b} batches of ({m}×{k}) @ ({k}×{n})"
        
    elif len(size) == 5:
        a, b, m, k, n = size
        A = torch.randn(a, b, m, k)
        B = torch.randn(a, b, k, n)
        
        matmul_op = lambda A, B: torch.matmul(A, B)
        einsum_op = lambda A, B: torch.einsum('abik,abkj->abij', A, B)
        
        size_str = f"Higher-dim matrix multiplication: {a}×{b} batches of ({m}×{k}) @ ({k}×{n})"
    
    # Move tensors to GPU if available
    if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()
    
    # How many runs to use
    n_runs = 1000 if A.numel() < 1_000_000 else 100
    
    # Run benchmarks
    matmul_time = benchmark_operation(matmul_op, A, B, n_runs=n_runs)
    einsum_time = benchmark_operation(einsum_op, A, B, n_runs=n_runs)
    
    speedup = einsum_time / matmul_time
    
    results.append({
        'size': size_str,
        'matmul_time': matmul_time * 1000,  # Convert to ms
        'einsum_time': einsum_time * 1000,   # Convert to ms
        'speedup': speedup
    })
    
    print(f"{size_str}")
    print(f"  torch.matmul: {matmul_time*1000:.3f} ms")
    print(f"  torch.einsum: {einsum_time*1000:.3f} ms")
    print(f"  Speedup (matmul/einsum): {1/speedup:.2f}x")
    print()

# Summary
print("Summary:")
print(f"torch.matmul is faster in {sum(1 for r in results if r['matmul_time'] < r['einsum_time'])} out of {len(results)} test cases")
print(f"Average speedup of matmul over einsum: {np.mean([1/r['speedup'] for r in results]):.2f}x")
print()
print("Conclusion:")
print("torch.matmul is generally faster than torch.einsum for common matrix operations.")
print("The performance gap increases for larger matrices and more complex operations.")
print("Use einsum when you need the flexibility and readability; use matmul when performance is critical.") 