# Mamba for macOS Apple Silicon

**Working Mamba implementation for macOS with pre-trained models**

This repository provides a **working Mamba implementation** optimized for **macOS Apple Silicon** with **downloaded pre-trained models** that actually work for text generation.

## 🎯 What Works

✅ **Pre-trained model loading** - 130M parameter Mamba model  
✅ **Text generation** - Quality text output with configurable parameters  
✅ **Apple Silicon optimized** - MPS acceleration support  
✅ **No triton dependency** - Graceful fallbacks for compatibility  
✅ **Comprehensive logging** - Detailed debugging and performance metrics  
✅ **Multiple interfaces** - Command line, interactive, demo, and benchmark modes  

## Quick Start

### Prerequisites
- macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.8+
- PyTorch with MPS support

### Installation & Setup

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio transformers einops

# 2. Clone and setup
git clone <this-repository>
cd mamba
pip install -e .

# 3. Download pre-trained model (493MB)
./download_mamba.sh

# 4. Run text generation
python run_mamba.py --prompt "The future of AI is" --max-length 100
```

## Usage Examples

### Basic Text Generation
```bash
# Simple generation
python run_mamba.py --prompt "Hello world" --max-length 50

# With custom parameters  
python run_mamba.py --prompt "Once upon a time" --max-length 100 --temperature 0.8
```

### Demo & Interactive Modes
```bash
# Interactive mode - chat with the model
python examples/07_downloaded_model_demo.py --interactive

# Demo mode - preset prompts showcase
python examples/07_downloaded_model_demo.py --demo

# Benchmark mode - performance testing
python examples/07_downloaded_model_demo.py --benchmark
```

### Performance Results (Apple Silicon M1)

| Mode | Length | Speed | Quality |
|------|--------|-------|---------|
| **Short** (20 tokens) | ~1s | 7.7 words/sec | ⭐⭐⭐⭐ |
| **Medium** (50 tokens) | ~6s | 4.2 words/sec | ⭐⭐⭐⭐ |
| **Long** (100 tokens) | ~22s | 2.6 words/sec | ⭐⭐⭐⭐⭐ |

## Model Details

- **Architecture**: Mamba SSM with 24 layers, 768 dimensions
- **Parameters**: 129M parameters  
- **Model Size**: 493MB download
- **Tokenizer**: GPT-NeoX (50k vocab)
- **Device**: MPS (Apple Silicon) with CPU fallback

## Features

### 🔧 Technical Features
- **Triton-free operation** - Works without CUDA dependencies
- **Graceful degradation** - Falls back to PyTorch when optimizations unavailable  
- **Memory efficient** - Optimized for Apple Silicon constraints
- **Robust error handling** - Comprehensive logging and fallbacks

### 🎮 User Features  
- **Multiple generation modes** - Interactive, demo, benchmark
- **Configurable parameters** - Temperature, length, sampling
- **Real-time performance metrics** - Speed and quality tracking
- **Beautiful output formatting** - Clear, readable results

## Repository Structure

```
├── run_mamba.py                     # 🎯 Main script - START HERE
├── download_mamba.sh                # 📥 Model download script  
├── examples/
│   └── 07_downloaded_model_demo.py  # 🎭 Interactive demo
├── tests/
│   └── test_downloaded_model.py     # 🧪 Working model tests
├── models/                          # 📁 Downloaded models (after setup)
│   ├── mamba-130m-config.json      
│   └── mamba-130m-model.bin        
└── mamba_ssm/                       # 🔧 Core implementation
    ├── models/mixer_seq_simple.py   # Model architecture
    ├── modules/mamba_simple.py      # Mamba layers  
    └── modules/block.py             # Building blocks
```

## Generated Text Examples

**🤖 AI Prediction:**
> "The future of artificial intelligence is not in limited solipsistic computing, but in densely-connected and much richer data. In the next decade, we may be able to take advantage of the huge amounts of new data..."

**📚 Creative Story:**  
> "Once upon a time, in a land far away, in a time of high prosperity, there lived one lonely woman, who was much respected among wolves. She resided at a rendezvous called Buguqrach..."

**💭 Philosophy:**
> "The key to happiness is to be able to make decisions quickly and without fear. When we are not in our own mind, we have very little control over what we choose to do..."

## Troubleshooting

### Common Issues

**❌ "Model files not found"**
```bash
# Run the download script
./download_mamba.sh
# Or check models/ directory exists
```

**❌ "Triton not available" warnings**  
✅ **This is normal!** - Our implementation works without triton

**❌ "MPS not available"**
```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
# Falls back to CPU automatically
```

**❌ Slow generation**
- First run is slower (model loading)
- Subsequent runs are faster (cached)
- Longer sequences take more time (expected)

## Testing

```bash
# Run comprehensive tests
python tests/test_downloaded_model.py

# Test basic functionality  
python run_mamba.py --prompt "Test" --max-length 10
```

## Advanced Usage

### Custom Model Directories
```bash
python run_mamba.py --model-dir /path/to/your/models --prompt "Custom model test"
```

### Performance Tuning
```bash
# Faster generation (less creative)
python run_mamba.py --prompt "Speed test" --temperature 0.1

# More creative (slower)  
python run_mamba.py --prompt "Creative test" --temperature 1.0
```

### Batch Processing
```python
from run_mamba import load_downloaded_model, generate_text

model, tokenizer = load_downloaded_model("models", "mps")
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

for prompt in prompts:
    text, time = generate_text(model, tokenizer, prompt)
    print(f"Generated: {text}")
```

## Technical Details

### Architecture Modifications
- **Forced Mamba1 usage** - Avoids triton-dependent Mamba2  
- **Disabled fused operations** - Uses PyTorch fallbacks
- **LayerNorm fallbacks** - When RMSNorm unavailable
- **Graceful import handling** - Continues despite missing optimizations

### Performance Optimizations  
- **MPS acceleration** - Apple Silicon GPU usage
- **Efficient tokenization** - Optimized input processing
- **Memory management** - Reduced peak usage
- **Caching strategies** - Faster subsequent runs

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Original architecture
- [State Spaces](https://github.com/state-spaces/mamba) - Reference implementation  
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html) - Apple Silicon acceleration

---

**Status**: ✅ **Production Ready** - Working text generation with pre-trained models on Apple Silicon