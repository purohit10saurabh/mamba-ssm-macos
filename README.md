# Mamba2MacOS - Apple Silicon Optimized

**[Mamba](https://arxiv.org/abs/2312.00752) & [Mamba2](https://arxiv.org/abs/2405.21060) SSM** implementation optimized for **macOS Apple Silicon** with MPS acceleration.

## About

This repository provides both **Mamba** (original) and **Mamba2** (latest) implementations, with a focus on the superior **Mamba2** architecture for high-speed text generation and sequence modeling on Apple Silicon devices.

This implementation leverages the **Metal Performance Shaders (MPS)** backend for PyTorch, ensuring optimal performance on Apple Silicon hardware.

**Mamba2** is a state-of-the-art SSM architecture that offers superior performance and efficiency for text generation and sequence modeling on Apple Silicon devices. It is designed to be faster and more efficient than the original Mamba architecture, and is the recommended architecture for new projects.

**Mamba** is the original Mamba architecture, which is stable and well-tested, and is recommended for existing projects and compatibility.

### 🚀 Key Features:
- **Dual architecture support** - Both Mamba (original) and Mamba2 (latest)
- **Apple Silicon optimized** with fast generation and MPS acceleration  
- **Mixed precision support** - FP32, FP16, BF16
- **Text generation** with configurable sampling
- **13 comprehensive tests** - All passing on Apple Silicon

### 🔄 Architecture Support:

| Model | Status | Performance Focus | Use Case |
|-------|--------|------------------|----------|
| **Mamba2** | ✅ **Primary** | **Fast generation** | **Recommended for new projects** |
| **Mamba** | ✅ **Legacy** | Stable & tested | Existing projects & compatibility |

**Recommendation**: Use **Mamba2** for new projects - it offers superior performance and efficiency.

## Quick Start

### Prerequisites
- macOS 12.3+ with Apple Silicon
- Python 3.8+
- PyTorch with MPS support

### Installation

```bash
pip install torch torchvision torchaudio
pip install einops transformers
git clone <this-repository-url>
cd mamba-ssm-macos
pip install -e .
```

## Usage

### Basic Usage

**Mamba2 (Recommended)**:
```python
import torch
from mamba_ssm.modules.mamba2_macos import Mamba2MacOS

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Mamba2MacOS(d_model=512, d_state=32, device=device)

x = torch.randn(2, 128, 512, device=device)
y = model(x)  # Output: torch.Size([2, 128, 512])
```

**Original Mamba (Legacy)**:
```python
import torch
from mamba_ssm.modules.mamba_simple import Mamba

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Mamba(d_model=512, d_state=16, device=device)

x = torch.randn(2, 128, 512, device=device)
y = model(x)  # Output: torch.Size([2, 128, 512])
```

### Text Generation

```bash
# Quick start - just works!
python examples/01_text_generation.py --prompt "Hello world"

# Or with module execution
python -m examples.01_text_generation --prompt "Hello world"
```

### Learning Examples

Follow the numbered examples for progressive learning:

```bash
python -m examples.01_text_generation    # 🎯 START HERE
python -m examples.02_basic_usage        # 🔧 Learn basics  
python -m examples.03_understanding_ssm  # 🧠 Understand theory
python -m examples.04_performance_analysis # ⚡ Benchmarks
python -m examples.05_mixed_precision    # 🔬 Precision modes
python -m examples.06_advanced_analysis  # 🧬 Advanced topics
```

## Model Configurations

| Size   | d_model | n_layer | Parameters |
|--------|---------|---------|------------|
| small  | 256     | 4       | ~14.5M     |
| medium | 512     | 8       | ~39.2M     |
| large  | 768     | 12      | ~87M+      |

## Mixed Precision

| Mode | Status | Stability |
|------|--------|-----------|
| **FP32** | ✅ Recommended | ✅ Stable |
| **FP16** | ⚠️ Caution | ⚠️ May be unstable |
| **BF16** | 🧪 Experimental | ⚠️ Limited support | 

**Recommendation**: Use FP32 for production on Apple Silicon.

## Testing

```bash
# Run all tests
python -m unittest discover tests -v

# Individual test suites
python -m unittest tests.test_mamba2_macos -v      # Core (11 tests)
python -m unittest tests.test_generation_macos -v  # Generation (2 tests)
python -m unittest tests.test_mamba_macos -v       # Legacy (2 tests)
```

## Repository Structure

```
├── mamba_ssm/modules/mamba2_macos.py    # Core implementation
├── examples/                            # 6 numbered examples
│   ├── 01_text_generation.py           # 🎯 START HERE
│   ├── 02_basic_usage.py              # Basic concepts
│   ├── 03_understanding_ssm.py         # SSM theory
│   ├── 04_performance_analysis.py      # Benchmarks
│   ├── 05_mixed_precision.py          # Precision analysis
│   └── 06_advanced_analysis.py        # Advanced topics
└── tests/                              # 13 comprehensive tests
    ├── test_mamba2_macos.py           # Core tests
    ├── test_generation_macos.py       # Generation tests
    └── test_mamba_macos.py            # Legacy tests
```

## Troubleshooting

### Common Issues

1. **MPS not available**: Check with `python -c "import torch; print(torch.backends.mps.is_available())"`
2. **Memory issues**: Start with small model size, reduce batch size
3. **Performance issues**: Ensure MPS is enabled, use appropriate model size
4. **Generation quality**: Adjust temperature, top_k, top_p, repetition_penalty.

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mamba-2 Paper](https://arxiv.org/abs/2405.21060)  
- [Original Mamba Implementation](https://github.com/state-spaces/mamba)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

---

**Status**: Production-ready with comprehensive Apple Silicon optimization ✅