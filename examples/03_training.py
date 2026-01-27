#!/usr/bin/env python3
"""
Simple Training Example

Demonstrates how to train a simple language model using Mamba modules.
Shows forward pass, loss calculation, backward pass, and optimizer step.
Supports both Mamba 1 (SSM) and Mamba 2 (SSD) architectures.
"""

import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CONFIG = OmegaConf.load(Path(__file__).with_suffix(".yaml"))
LAYER_CONFIG = CONFIG.layer_config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LearnableSequenceDataset(Dataset):
    def __init__(
        self, *, size: int, seq_len: int, vocab_size: int, task: str, base_seed: int
    ):
        self._size, self._seq_len, self._vocab_size = size, seq_len, vocab_size
        self._task, self._base_seed = task, base_seed

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(self._base_seed + idx)
        input_ids = torch.randint(
            0, self._vocab_size, (self._seq_len,), device=DEVICE, generator=generator
        )
        if self._task == "shift":
            labels = torch.roll(input_ids, shifts=1, dims=0)
            labels[0] = 0
        else:
            labels = input_ids.clone()
        return input_ids, labels


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class SimpleMambaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=LAYER_CONFIG["d_conv"],
                    expand=LAYER_CONFIG["expand"],
                    device=DEVICE,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(DEVICE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class SimpleMamba2LM(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, d_state: int, n_layers: int, headdim: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=LAYER_CONFIG["d_conv"],
                    expand=LAYER_CONFIG["expand"],
                    headdim=headdim,
                    ngroups=LAYER_CONFIG["ngroups"],
                    chunk_size=LAYER_CONFIG["chunk_size"],
                    device=DEVICE,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(DEVICE)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(
    model: nn.Module, dataloader: Iterable, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).float().sum().item()
            total_tokens += labels.numel()
            total_loss += loss.item()
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_tokens)
    return avg_loss, accuracy


def train(model: nn.Module, model_name: str, cfg: DictConfig) -> float:
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_size = cfg.batch_size * cfg.num_steps
    train_dataset = LearnableSequenceDataset(
        size=train_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        task=cfg.task,
        base_seed=cfg.seed,
    )
    eval_dataset = LearnableSequenceDataset(
        size=cfg.batch_size * cfg.eval_batches,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        task=cfg.task,
        base_seed=cfg.seed + train_size,
    )
    train_loader = create_dataloader(train_dataset, batch_size=cfg.batch_size)
    eval_loader = create_dataloader(eval_dataset, batch_size=cfg.batch_size)

    last_accuracy = 0.0
    for step, (input_ids, labels) in tqdm(
        enumerate(train_loader, start=1),
        desc=f"Training {model_name.upper()}",
        total=cfg.num_steps,
    ):
        loss = train_step(model, input_ids, labels, optimizer, criterion)
        if step % cfg.eval_every_steps == 0 or step == 1:
            test_loss, accuracy = evaluate(model, eval_loader, criterion)
            last_accuracy = accuracy
            tqdm.write(
                f"Step {step}, Loss: {loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}"
            )
    if last_accuracy == 0.0:
        _, last_accuracy = evaluate(model, eval_loader, criterion)
    return last_accuracy


def print_report(cfg: DictConfig, results: dict) -> None:
    print("\n📊 Training Report")
    print("=" * 50)
    print(f"Task: {cfg.task} | Steps: {cfg.num_steps} | Batch size: {cfg.batch_size}")
    print(
        "Dataset sizes: "
        f"train={cfg.batch_size}×{cfg.num_steps}, "
        f"eval={cfg.batch_size}×{cfg.eval_batches}"
    )
    print(f"\nFinal test accuracy after training for {cfg.num_steps} steps:")
    for name, accuracy in results.items():
        title = CONFIG.models[name].title
        print(f"- {title}: {accuracy:.2%}")
    print("\n💡 Both architectures support full training with backpropagation!")


def main() -> None:
    print("🎓 Simple Training Example")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    set_global_seed(CONFIG.training.seed)

    print(f"Task: {CONFIG.training.task}")
    print(f"Training steps: {CONFIG.training.num_steps}")

    results = {}
    for name, config in CONFIG.models.items():
        print(f"\n--- Training {config.title} ---")
        model = instantiate(config.model)
        results[name] = train(model=model, model_name=name, cfg=CONFIG.training)

    print_report(CONFIG.training, results)


if __name__ == "__main__":
    main()
