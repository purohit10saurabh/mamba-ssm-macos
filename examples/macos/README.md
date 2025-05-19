# Mamba2 macOS Examples

This directory contains specialized examples and demos for running Mamba2 on macOS. The main example script is located in the parent directory (`run_mamba2_macos_example.py`).

## File Organization

- `run_mamba2_macos_example.py` (in parent directory)
  - Main example script with comprehensive text generation and benchmarking
  - Includes proper tokenizer support and advanced generation features
  - Use this for most use cases

- `selective_scan_demo.py`
  - Demonstrates the selective scan operation
  - Useful for understanding the core SSM mechanism
  - Shows how the selective scan works without CUDA/Triton

- `minimal_ssm_demo.py`
  - Shows a minimal SSM implementation
  - Good for learning the basics of state space models
  - Includes simple sequential scan implementation

- `basic_demo.py`
  - Basic Mamba functionality demo
  - Shows core operations without advanced features
  - Good starting point for understanding Mamba

- `benchmark.py`
  - Detailed benchmarking tools
  - Includes performance comparisons between CPU and MPS
  - Useful for measuring model performance

## Usage

For most use cases, start with the main example script:

```bash
# Basic text generation
python ../run_mamba2_macos_example.py --prompt "Once upon a time"

# Run benchmark
python ../run_mamba2_macos_example.py --benchmark --model-size large
```

For specialized demos, run the individual scripts:

```bash
# Run selective scan demo
python selective_scan_demo.py

# Run minimal SSM demo
python minimal_ssm_demo.py

# Run basic demo
python basic_demo.py

# Run benchmarks
python benchmark.py
```

## Notes

- All examples are designed to work on macOS, particularly Apple Silicon
- MPS (Metal Performance Shaders) acceleration is used when available
- The examples use reference implementations where CUDA/Triton is not available
- See the main example script for the most up-to-date and complete implementation 