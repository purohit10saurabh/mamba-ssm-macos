# Running Mamba SSM on macOS Apple Silicon

This document explains how to run Mamba SSM on macOS Apple Silicon (M1/M2/M3) machines, which don't have direct support for CUDA extensions and Triton.

## Installation

To install Mamba SSM on macOS Apple Silicon, use the following command:

```bash
CUDA_HOME="" MAMBA_SKIP_CUDA_BUILD=TRUE pip install -e .
```

This will build Mamba with CPU-only fallbacks for functionality that usually requires CUDA or Triton.

## Limitations

On macOS Apple Silicon, the following limitations apply:

1. CUDA extensions are not available, so the selective scan operation uses a slower reference implementation.
2. Triton is not available, so the layernorm implementations use slower PyTorch fallbacks.
3. Some modules that strictly require Triton (like Mamba2) are not available.

## Running Tests

To run tests specifically designed for macOS:

```bash
# Run the basic macOS tests
python tests/test_macos.py

# Run the generation test for macOS
python tests/test_generation_macos.py
```

To run all compatible tests:

```bash
python -m unittest discover tests
```

## Performance Considerations

Due to the CPU-only fallbacks, performance will be significantly slower compared to CUDA-accelerated systems. This implementation is primarily for development, testing, and educational purposes on macOS.

You can benchmark performance on macOS using the provided benchmark script:

```bash
python examples/benchmark_macos.py
```

This script compares CPU and MPS (Metal Performance Shaders) performance when available.

## Example Scripts

Several macOS-compatible example scripts are provided:

```bash
# Run basic Mamba functionality
python examples/run_mamba_basic.py

# Run minimal Mamba implementation
python examples/run_mamba_minimal.py

# Run selective scan demo
python examples/run_mamba_selective.py

# Run Mamba2-like model (compatible with macOS)
python examples/run_mamba2_macos.py
```

## Supported Features

The following features are supported on macOS Apple Silicon:

1. Basic Mamba forward and backward passes
2. RMSNorm and LayerNorm operations (using PyTorch fallbacks)
3. Selective scan operation (using the reference implementation)
4. Simple models using the Mamba architecture
5. Text generation with variable lengths
6. MPS (Metal Performance Shaders) acceleration for some operations

## Troubleshooting

If you encounter issues, ensure that:

1. You're using the CUDA_HOME="" MAMBA_SKIP_CUDA_BUILD=TRUE environment variables during installation
2. You have PyTorch installed with MPS support (for Apple Silicon acceleration)
3. Your code doesn't explicitly import modules that require Triton (e.g., Mamba2)

A native implementation of Mamba SSM (State Space Model) optimized for macOS Apple Silicon (M1/M2/M3) devices. This repository provides CPU and MPS (Metal Performance Shaders) implementations of Mamba, enabling efficient inference and training on Apple Silicon without CUDA dependencies.

Key features:
• Native macOS support with MPS acceleration
• CPU-optimized selective scan operations
• PyTorch-based implementations of core components
• Comprehensive test suite for macOS compatibility
• Example scripts for basic Mamba functionality
• Benchmarking tools for performance comparison

Perfect for developers, researchers, and ML practitioners who want to run Mamba models on Apple Silicon devices. 