# Mamba for macOS Apple Silicon

![Mamba](assets/selection.png "Selective State Space")
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

![Mamba-2](assets/ssd_algorithm.png "State Space Dual Model")
> **Transformers are SSMs: Generalized Models and Efficient Algorithms**\
>     **Through Structured State Space Duality**\
> Tri Dao*, Albert Gu*\
> Paper: https://arxiv.org/abs/2405.21060

## About

This repository provides an implementation of Mamba SSM (State Space Model) optimized for macOS Apple Silicon (M1/M2/M3) devices. It enables efficient inference and training on Apple Silicon without CUDA dependencies, making Mamba accessible to macOS users.

Key features:
- macOS support with MPS (Metal Performance Shaders) acceleration for GPU operations
- CPU-optimized selective scan operations with SIMD optimizations
- PyTorch-based implementations with MPS backend support
- Comprehensive test suite for macOS compatibility
- Example scripts for basic Mamba functionality
- Text generation example with configurable model sizes
- Benchmarking tools for performance comparison between CPU and MPS

This implementation is based on the original [Mamba](https://github.com/state-spaces/mamba) architecture, which showed promising performance on information-dense data such as language modeling, where previous subquadratic models fall short of Transformers.

## Quick Start

1. Ensure you have the prerequisites:
   - macOS 12.3+ with Apple Silicon (M1/M2/M3)
   - Python 3.8+
   - Xcode Command Line Tools
   - PyTorch with MPS support
   - transformers (for text generation example)

2. Install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

3. Install Mamba SSM:
```bash
CUDA_HOME="" MAMBA_SKIP_CUDA_BUILD=TRUE pip install -e .
```

## Model Sizes

The implementation provides three model size configurations:

| Size   | d_model | n_layer | d_state | Parameters (approx) |
|--------|---------|---------|---------|-------------------|
| small  | 256     | 4       | 16      | ~1M               |
| medium | 512     | 8       | 32      | ~8M               |
| large  | 1024    | 12      | 64      | ~32M              |

Note: These are untrained models. For production use, you would need to load pre-trained weights.

## Usage

### Basic Mamba Block

The main module provides a macOS-compatible implementation of the Mamba architecture block:

```python
import torch
from mamba_ssm import Mamba

# Choose device (CPU or MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to(device)
model = Mamba(
    d_model=dim,    # Model dimension d_model
    d_state=16,     # SSM state expansion factor
    d_conv=4,       # Local convolution width
    expand=2,       # Block expansion factor
).to(device)
y = model(x)
assert y.shape == x.shape
```

### Example Scripts

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

# Run text generation example
python examples/run_generation_macos.py --prompt "Your prompt here" --model-size small
```

### Text Generation Example

The text generation example (`run_generation_macos.py`) provides a simple interface for generating text:

```bash
# Basic usage
python examples/run_generation_macos.py --prompt "Your prompt here"

# Advanced options
python examples/run_generation_macos.py \
    --prompt "Your prompt here" \
    --model-size medium \
    --temperature 0.8 \
    --top-p 0.9 \
    --repetition-penalty 1.2 \
    --max-length 200
```

Available options:
- `--model-size`: Choose from "small", "medium", "large" (default: small)
- `--prompt`: Input text to start generation (required)
- `--max-length`: Maximum length of generated text (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling parameter (default: 0.9)
- `--repetition-penalty`: Penalty for repeating tokens (default: 1.2)

Note: The generation example uses randomly initialized models. For better results, you would need to load pre-trained weights.

## Limitations

On macOS Apple Silicon, the following limitations apply:

1. CUDA extensions are not available, so the selective scan operation uses a slower reference implementation
2. Triton is not available, so the layernorm implementations use slower PyTorch fallbacks
3. Some modules that strictly require Triton (like Mamba2) are not available
4. Performance will be significantly slower compared to CUDA-accelerated systems
5. Memory usage may be higher due to CPU-based implementations
6. Some advanced features from the original Mamba implementation may not be available
7. Text generation uses untrained models by default

This implementation is primarily for development, testing, and educational purposes on macOS.

## Running Tests

To run tests specifically designed for macOS:

```bash
# Run the basic macOS tests
python tests/test_macos.py

# Run the generation test for macOS
python tests/test_generation_macos.py

# Run all compatible tests
python -m unittest discover tests

# Run performance benchmarks
python tests/benchmark_macos.py
```

## Performance Tips

1. Use MPS backend when available for better performance
2. Monitor memory usage as CPU implementations may use more memory
3. For large models, consider using gradient checkpointing
4. Use appropriate batch sizes based on your available memory
5. Enable MPS fallback to CPU when needed for better stability
6. For text generation, start with the small model size and increase if needed

## Troubleshooting

### Common Issues

1. MPS backend not available:
   - Update to latest PyTorch version
   - Ensure macOS version is 12.3 or later
   - Check if MPS is enabled in PyTorch

2. Memory issues:
   - Reduce batch size
   - Use smaller model configurations
   - Monitor memory usage with Activity Monitor

3. Slow performance:
   - Ensure MPS is enabled and working
   - Check if running on CPU instead of MPS
   - Consider using smaller models for development

4. Installation errors:
   - Verify Xcode Command Line Tools installation
   - Check Python version compatibility
   - Ensure proper environment variables are set

5. Text generation issues:
   - Start with small model size
   - Adjust temperature and top-p parameters
   - Use shorter prompts initially
   - Note that models are untrained by default

### Environment Setup

If you encounter issues, ensure that:

1. You're using the `CUDA_HOME="" MAMBA_SKIP_CUDA_BUILD=TRUE` environment variables during installation
2. You have PyTorch installed with MPS support (for Apple Silicon acceleration)
3. Your code doesn't explicitly import modules that require Triton (e.g., Mamba2)
4. Xcode Command Line Tools are installed
5. You're using a compatible Python version (3.8+)
6. Your macOS version supports MPS (macOS 12.3+)

## Additional Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/)
- [Original Mamba Repository](https://github.com/state-spaces/mamba)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)