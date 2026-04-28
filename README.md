# Mamba SSM for macOS Apple Silicon

**[Mamba 1](https://arxiv.org/abs/2312.00752) and [Mamba 2](https://arxiv.org/abs/2405.21060) State Space Models for Apple Silicon**

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-blue?logo=apple)](https://developer.apple.com/mac/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-orange?logo=pytorch)](https://pytorch.org)
[![PyPI](https://img.shields.io/pypi/v/mamba-ssm-macos)](https://pypi.org/project/mamba-ssm-macos/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Training and inference of Mamba 1 & 2 on Apple Silicon with MPS acceleration. Works without CUDA/Triton.

## Installation

```bash
pip install mamba-ssm-macos
```

Prerequisites: macOS 12.3+ with Apple Silicon, Python 3.10+, 8GB+ RAM recommended.

## Quickstart

```python
import torch
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel, generate_text_with_model, get_device

device = get_device()
model_name = "state-spaces/mamba-130m"  # or "state-spaces/mamba2-130m"
model = MambaLMHeadModel.from_pretrained(model_name, device=device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
text = generate_text_with_model(model, tokenizer, "Once upon a time", device, max_length=50, temperature=0.8)
print(text)
```

The model is downloaded from Hugging Face Hub on first run and cached under `~/.cache/huggingface/`.

## Training

```python
import torch
from torch import nn
from mamba_ssm.modules.mamba2 import Mamba2

model = nn.Sequential(nn.Embedding(1000, 128), *[Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256, device='mps') for _ in range(2)], nn.LayerNorm(128), nn.Linear(128, 1000, bias=False)).to('mps')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for input_ids, labels in dataloader:
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
```

See `examples/03_training.py` for a complete training example.

## Examples

Clone the repo to run examples:

```bash
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync
python -m examples.01_core_modules  # Core modules usage
python -m examples.02_text_generation --interactive  # Text generation
python -m examples.03_training  # Training
```

## Troubleshooting

**"MPS not available"** — Check with `python -c "import torch; print(torch.backends.mps.is_available())"`. The library falls back to CPU automatically.

**Slow first run** — Initial `from_pretrained` downloads ~500MB of weights from Hugging Face Hub. Subsequent runs use the local cache.

## Development

```bash
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync --extra dev
make test
```

Available `make` targets: `test`, `test-unit`, `test-integration`, `test-quick`, `format`, `format-check`, `clean`. See the [Makefile](Makefile) for details.

## Citation

Also available via GitHub's "Cite this repository" button ([CITATION.cff](CITATION.cff)).

```bibtex
@software{purohit2026mamba_ssm_macos,
  title={Mamba SSM for macOS Apple Silicon},
  author={Purohit, Saurabh},
  year={2026},
  url={https://github.com/purohit10saurabh/mamba-ssm-macos}
}
```

<details>
<summary>Original Mamba papers</summary>

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{dao2024transformers,
  title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2405.21060},
  year={2024}
}
```

</details>

## References

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Original implementation
- [state-spaces/mamba1-130m](https://huggingface.co/state-spaces/mamba-130m) — Mamba 1 130M pre-trained model
- [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) — Mamba 2 130M pre-trained model

## Contributing

Feel free to [open an issue](https://github.com/purohit10saurabh/mamba-ssm-macos/issues) or [submit a PR](https://github.com/purohit10saurabh/mamba-ssm-macos/pulls). See [Development](#development) for the local setup.

## License

Apache 2.0 — see [LICENSE](LICENSE).