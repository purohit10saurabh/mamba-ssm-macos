# Mamba SSM for macOS Apple Silicon

**[Mamba 1](https://arxiv.org/abs/2312.00752) and [Mamba 2](https://arxiv.org/abs/2405.21060) State Space Models for Apple Silicon**

[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-blue?logo=apple)](https://developer.apple.com/mac/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Training and inference of Mamba 1 & 2 on Apple Silicon with MPS acceleration. Works without CUDA/Triton. Supports CLI, Python API, and interactive demos.

## Quick Start

```bash
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync                                     # or: pip install -r requirements.txt

python -m scripts.download_models mamba1    # Mamba 1 (493MB)
python -m scripts.download_models mamba2    # Mamba 2 (493MB)

make run-mamba1                             # Quick Mamba 1 demo
make run-mamba2                             # Quick Mamba 2 demo
```

**Prerequisites:** macOS 12.3+ with Apple Silicon, Python 3.10+, 8GB+ RAM recommended.

## Usage

### Text Generation
```bash
python -m scripts.run_models mamba1 --prompt "The future of AI" --max-length 50
python -m scripts.run_models mamba2 --prompt "The future of AI" --max-length 30
python -m scripts.run_models mamba1 --prompt "Once upon a time" --temperature 0.8
python -m examples.02_text_generation --interactive
```

### Examples
```bash
python -m examples.01_core_modules     # Core modules usage
python -m examples.02_text_generation  # Text generation demo
python -m examples.03_training         # Training example
```

### Makefile Commands
```bash
make download-models   # Download both models
make run-mamba1        # Quick Mamba 1 demo
make run-mamba2        # Quick Mamba 2 demo
make test-quick        # Fast integration test
make test              # Full test suite
```

## Training

See `examples/03_training.py` for a full example. Snippet:

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

## Repository Structure

```
mamba-ssm-macos/
├── mamba_ssm/              # Core library (models, modules, ops, utils)
├── scripts/                # download_models.py, run_models.py
├── tests/                  # unit/, integration/, run_unit_tests.py
├── examples/               # 01_core_modules, 02_text_generation, 03_training
├── Makefile
└── pyproject.toml
```

## Troubleshooting

**"Model files not found"** — Run `make download-models` or `python -m scripts.download_models mamba1|mamba2`.

**"MPS not available"** — Check with `python -c "import torch; print(torch.backends.mps.is_available())"`. Falls back to CPU automatically.

**Import errors** — Use module syntax: `python -m examples.02_text_generation`.

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
- [state-spaces/mamba1-130m](https://huggingface.co/state-spaces/mamba1-130m) — Mamba 1 130M Pre-trained model
- [state-spaces/mamba2-130m](https://huggingface.co/state-spaces/mamba2-130m) — Mamba 2 130M Pre-trained model

## Contributing

Contributions are welcome — bug fixes, performance improvements, docs, and new features. Open an issue or submit a PR.

```bash
git clone https://github.com/purohit10saurabh/mamba-ssm-macos.git
cd mamba-ssm-macos
uv sync --extra dev
make test
```

## License

Apache 2.0 — see [LICENSE](LICENSE).