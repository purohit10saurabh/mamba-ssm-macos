# 🐍 Mamba for macOS Apple Silicon

**High-performance [Mamba 1](https://arxiv.org/abs/2312.00752) & [Mamba 2](https://arxiv.org/abs/2405.21060) implementation optimized for Apple Silicon with official pre-trained models**

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-blue?logo=apple)](https://developer.apple.com/mac/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

- **Mamba 1 & 2 Support** - Inference of both architectures with pretrained models from Hugging Face
- **Text Generation** - Coherent, contextual text generation 
- **Apple Silicon Support** - MPS acceleration for M1/M2/M3/M4
- **Dependency Management** - Works without CUDA/Triton requirements
- **Error Handling** - Robust error handling and fallbacks for both architectures
- **Multiple Interfaces** - CLI, Python API, interactive demos  

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync

# 2. Download models 
uv run python -m scripts.download_models mamba1    # Mamba 1 (493MB)
uv run python -m scripts.download_models mamba2    # Mamba 2 (493MB) 

# 3. Generate text immediately  
make run-mamba1                              # Quick Mamba 1 demo
make run-mamba2                              # Quick Mamba 2 demo
uv run python -m examples.01_core_modules # Core modules usage
uv run python -m examples.02_text_generation # Text generation demo
uv run python -m examples.03_training # Training example
```

## Table of Contents

- [Architecture Comparison](#architecture-comparison)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Performance](#performance)
- [Generated Examples](#generated-examples)
- [Repository Structure](#repository-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [References](#references)
- [Contributing](#contributing)

## Architecture Comparison

| Feature | Mamba 1 | Mamba 2 |
|---------|---------|------------|
| **Architecture** | SSM (Selective State Space) | SSD (State Space Dual) |
| **Training Speed** | Standard | ~2x faster |
| **State Dimension** | 16 | 128 (8x larger) |
| **Multi-head** | No | Yes (via ngroups) |
| **Memory Efficiency** | Good | Better |
| **Generation Quality** | High | Higher |
| **Model Size** | 129M params | 129M params |

## Installation

### Prerequisites
- **macOS 12.3+** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.8+**
- **8GB+ RAM** recommended

### Setup
```bash
# Clone repository
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos

# Install dependencies (includes PyTorch with MPS support)
uv sync

# Verify MPS support
uv run python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

### Download Models

#### Both Models (Recommended)
```bash
make download-models  # Downloads both Mamba 1 & 2
```

#### Individual Models
```bash
uv run python -m scripts.download_models mamba1  # Mamba 1 (original)
uv run python -m scripts.download_models mamba2  # Mamba 2 (latest)
```

## Usage Examples

### Mamba 2 (Latest)

#### Quick Test
```bash
uv run python -m examples.02_text_generation --interactive  # Try both models
uv run python -m examples.02_text_generation --show-structure  # See organization
```

#### Makefile Commands
```bash
make run-mamba1         # Quick Mamba 1 demo
make run-mamba2         # Quick Mamba 2 demo  
make test-quick         # Fast integration test
make show-structure     # Show directory layout
```

### Command Line Generation

#### Mamba 1 & 2 via Scripts
```bash
# Basic generation
uv run python -m scripts.run_models mamba1 --prompt "The future of AI" --max-length 50
uv run python -m scripts.run_models mamba2 --prompt "The future of AI" --max-length 30

# Custom parameters
uv run python -m scripts.run_models mamba1 --prompt "Once upon a time" --temperature 0.8
```

### Learning Examples
```bash
uv run python -m examples.01_core_modules      # Core modules usage
uv run python -m examples.02_text_generation  # Text generation demo
uv run python -m examples.03_training        # Training example
```

## Performance

### Apple Silicon Results (for M1)

| Model | Loading | Generation | Memory | Quality |
|-------|---------|------------|--------|---------|
| **Mamba 1** | ~1.0s | 3-8 tok/s | ~2GB | Good |
| **Mamba 2** | ~1.0s | 3-6 tok/s | ~2GB | Better |

### Benchmark Results
```bash
# Test performance
make test-quick
```

**Mamba 2 Advantages:**
- Similar loading speed
- Better context understanding (d_state=128 vs 16)
- Higher quality output (SSD architecture)
- More efficient training (~2x faster during training)

## Generated Examples

### Mamba 2 (SSD Architecture)
```
"The future of artificial intelligence is a big topic in the field of artificial intelligence."

"Once upon a time, there was a man named John."

"Python is a programming language that is used to create and manipulate objects."

"The capital of France is a city of the French, and the"
```

### Mamba 1 (SSM Architecture)
```
"The future of AI is not in limited solipsistic computing, but in densely-connected 
    and much richer data. In the next decade, we may be able to take advantage..."

"Once upon a time, in a land far away, there lived one lonely woman, who was 
    much respected among wolves. She resided at a rendezvous called Buguqrach..."
```

## Repository Structure

```
mamba-ssm-macos/
├── 📦 src/mamba_macos/               # Core library
│   ├── __init__.py                   # Package exports
│   ├── utils.py                      # Device, tokenizer, generation
│   └── models.py                     # Model loading & preparation
│
├── 🔧 scripts/                       # Utility scripts
│   ├── download_models.py            # Download both models
│   └── run_models.py                 # Run models with arguments
│
├── 🧪 tests/                         # Test suite  
│   ├── unit/                         # Component-level tests
│   │   ├── test_mamba_macos.py       # Mamba 1 unit tests
│   │   ├── test_mamba2_macos.py      # Mamba 2 unit tests
│   │   └── test_generation_macos.py  # Generation tests
│   ├── integration/                  # End-to-end tests
│   │   └── test_unified_system.py    # Complete workflow tests
│   └── run_all_tests.py              # Test runner
│
├── 📚 examples/                       # Curated examples
│   ├── 01_core_modules.py             # Core modules usage (Mamba 1 & 2)
│   ├── 02_text_generation.py          # Text generation with pretrained models
│   └── 03_training.py                 # Simple training example
│
├── ⚙️ config/                        # Configuration files
│   └── pyproject.toml                # Python project config
│
├── mamba_ssm/                        # Core implementation
│   ├── models/                       # Model architectures
│   ├── modules/                      # Model modules
│   ├── ops/                          # Operations (CPU/Triton fallbacks)
│   ├── utils/                        # Utilities
│   └── distributed/                  # Distributed utilities
│
├── 📋 Makefile                       # Development commands
├── 📋 pyproject.toml                 # Project configuration
├── 📋 uv.toml                        # UV configuration
└── 📖 README.md                      # This file
```

## Advanced Usage

### Custom Model Configuration
```python
# Mamba 2 custom config
config = MambaConfig(
    d_model=768,
    n_layer=24,
    d_state=128,           # Larger state space
    headdim=64,           # Head dimension
    expand=2,             # Expansion factor
    ssm_cfg={"layer": "Mamba2", "d_state": 128},
    vocab_size=50288
)

# Mamba 1 custom config  
config = MambaConfig(
    d_model=768,
    n_layer=24,
    d_state=16,           # Smaller state space
    ssm_cfg={"layer": "Mamba1"},
    vocab_size=50280
)
```

### Batch Processing
```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
for prompt in prompts:
    # Process each prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    print(tokenizer.decode(outputs[0]))
```

### Fine-tuning Setup
```python
# Prepare for fine-tuning
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Your training loop here
for batch in dataloader:
    outputs = model(batch['input_ids'], labels=batch['labels'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

## Troubleshooting

### Common Issues

#### ❌ "Model files not found"
```bash
# Download models using new structure
make download-models                         # Both models
uv run python -m scripts.download_models mamba1    # Mamba 1 only
uv run python -m scripts.download_models mamba2    # Mamba 2 only
```

#### ❌ "MPS not available"
```bash
# Check MPS support
uv run python -c "import torch; print(torch.backends.mps.is_available())"

# If false, model will automatically use CPU
```

#### ❌ Import errors
```bash
# Use module structure
uv run python -m examples.02_text_generation
```

#### ❌ Slow generation
- ✅ **First run is slower** (model loading + compilation)
- ✅ **Use shorter prompts** for testing
- ✅ **Close other apps** to free memory
- ✅ **Check Activity Monitor** for memory usage

### Expected Warnings (Safe to Ignore)
```
UserWarning: selective_scan_cuda module is not available
UserWarning: Triton is not available  
```
These are expected - we use optimized PyTorch fallbacks.

### Getting Help
1. 📖 **Read the docs**: See this README for installation, usage examples, and troubleshooting
2. 🧪 **Run tests**: `make test-quick` or `make test` 
3. 🔍 **Check examples**: `uv run python -m examples.02_text_generation --show-structure`
4. 🐛 **Report issues**: Create GitHub issue with error details

## Learning Path

### Start Here (3 Steps)
```bash
# 1. Download models
make download-models

# 2. Test basic functionality  
make run-mamba1

# 3. Explore interactively
uv run python -m examples.02_text_generation
```

### Build Something
```bash
# Use Python API
uv run python -m examples.01_core_modules

# Custom generation
uv run python -m scripts.run_models mamba1 --prompt "Your text"
```

## Technical Details

### Mamba 2 Implementation Highlights
- **State Space Dual (SSD)** architecture from official state-spaces/mamba
- **Stable cumulative scan** for numerical stability
- **Multi-head processing** with ngroups design × 64 headdim  
- **Larger state space** (d_state=128) for better memory
- **Einsum operations** for efficient tensor computations
- **MPS optimization** for Apple Silicon acceleration

### Mamba 1 Implementation Highlights  
- **Selective State Space Model (SSM)** architecture
- **Triton-free operation** with PyTorch fallbacks
- **Graceful degradation** when optimizations unavailable
- **Memory efficient** selective scan implementation
- **Compatible** with original mamba-130m weights

## What's Next?

### Immediate Use
1. **Download model**: Choose Mamba 1 or 2
2. **Test functionality**: Run example scripts
3. **Try your prompts**: Experiment with generation
4. **Read examples**: Learn from provided demos

### Advanced Projects
1. **Fine-tune models**: Train on your data
2. **Build applications**: Use as text generation backend
3. **Contribute**: Improve implementation or docs
4. **Research**: Experiment with architectures

## Citation

If you use this implementation in your research or project, please cite it. GitHub will automatically recognize the citation from the [CITATION.cff](CITATION.cff) file.

**BibTeX format:**
```bibtex
@software{purohit2026mamba_macos,
  title={Mamba for macOS Apple Silicon: Optimized Implementation},
  author={Purohit, Saurabh and Dao, Tri and Gu, Albert},
  year={2026},
  url={https://github.com/purohit10saurabh/mamba-ssm-macos}
}
```

**APA format:**
> Purohit, S., Dao, T., & Gu, A. (2026). Mamba for macOS Apple Silicon: Optimized implementation [Computer software]. https://github.com/purohit10saurabh/mamba-ssm-macos

## References

### Papers

#### Mamba 1: Linear-Time Sequence Modeling
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023},
  url={https://arxiv.org/abs/2312.00752}
}
```

#### Mamba 2: Structured State Space Duality
```bibtex
@article{dao2024transformers,
  title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2405.21060},
  year={2024},
  url={https://arxiv.org/abs/2405.21060}
}
```

### Official Implementations

- **🔗 Mamba 1 & 2**: [state-spaces/mamba](https://github.com/state-spaces/mamba) - Original PyTorch implementation
- **🤗 Mamba 2 Model**: [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) - Pre-trained weights
- **🔬 State Space Models**: [state-spaces/s4](https://github.com/state-spaces/s4) - Foundational SSM research

### Related Work

- **Selective State Spaces**: [Gu et al., 2022](https://arxiv.org/abs/2208.04933) - S4 foundation
- **Hungry Hungry Hippos**: [Fu et al., 2023](https://arxiv.org/abs/2212.14052) - H3 architecture  
- **Apple Silicon**: [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html) - Metal Performance Shaders

## Contributing

We welcome contributions! Areas for improvement:

- 🐛 **Bug fixes**: Report and fix issues
- 📚 **Documentation**: Improve guides and examples  
- ⚡ **Performance**: Optimize for specific hardware
- 🆕 **Features**: Add new capabilities
- 🧪 **Testing**: Expand test coverage

### Development Setup
```bash
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync --extra dev
make test
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) file.

---

**Optimized for Apple Silicon • Pure Python • High-Performance**

*Start with `uv run python -m examples.01_core_modules` and explore from there!* ⬆️