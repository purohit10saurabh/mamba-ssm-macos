#!/usr/bin/env python3
"""
Simple Training Example

Demonstrates how to train a simple language model using Mamba modules.
Shows forward pass, loss calculation, backward pass, and optimizer step.
Supports both Mamba 1 (SSM) and Mamba 2 (SSD) architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS
from mamba_ssm.modules.mamba_simple import Mamba

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class SimpleMambaLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, d_state=16, n_layers=2, device=DEVICE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2, device=device)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(device)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class SimpleMamba2LM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, d_state=64, n_layers=2, headdim=64, device=DEVICE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba2MacOS(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                headdim=headdim,
                ngroups=1,
                chunk_size=256,
                device=device
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(device)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def create_learnable_data(batch_size=4, seq_len=32, vocab_size=100, task="shift"):
    """
    Create data with elementary transformations to verify model learning.
    
    Tasks:
    - "shift": labels[i] = input_ids[i-1] (previous token prediction)
    - "copy": labels[i] = input_ids[i] (copy current token)
    """
    if task not in ["shift", "copy"]:
        raise ValueError(f"Invalid task: {task}")
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
    if task == "shift":
        labels = torch.roll(input_ids, shifts=1, dims=1)
        labels[:, 0] = 0
    else:
        labels = input_ids.clone()
    return input_ids, labels

def train_step(model, input_ids, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, input_ids, labels, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    return loss.item(), accuracy


def train_mamba1(task, num_steps):
    print("\n🔷 Training Mamba 1 (SSM Architecture)")
    print("=" * 50)

    vocab_size = 100
    d_model = 128
    d_state = 16
    n_layers = 2
    batch_size = 8
    seq_len = 32

    print(f"📊 Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  d_state: {d_state}")
    print(f"  Layers: {n_layers}")
    task_desc = {
        "shift": "labels[i] = input[i-1]",
        "copy": "labels[i] = input[i]"
    }
    print(f"  Task: {task} ({task_desc.get(task, 'unknown')})")

    model = SimpleMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        device=DEVICE
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔄 Training for {num_steps} steps...")
    initial_loss = None
    for step in range(1, num_steps + 1):
        input_ids, labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
        loss = train_step(model, input_ids, labels, optimizer, criterion)
        
        if step == 1:
            initial_loss = loss
        
        if step % 5 == 0 or step == 1:
            test_input, test_labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
            test_loss, accuracy = evaluate(model, test_input, test_labels, criterion)
            print(f"  Step {step:2d}/{num_steps}: Loss = {loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.2%}")

    if initial_loss is not None:
        final_test_input, final_test_labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
        final_loss, final_accuracy = evaluate(model, final_test_input, final_test_labels, criterion)
        improvement = initial_loss - final_loss
        print(f"\n📈 Learning Progress:")
        print(f"  Initial Loss: {initial_loss:.4f}")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/initial_loss*100:.1f}% reduction)")
        print(f"  Final Accuracy: {final_accuracy:.2%}")

    print("✅ Mamba 1 training completed!")


def train_mamba2(task, num_steps):
    print("\n🔶 Training Mamba 2 (SSD Architecture)")
    print("=" * 50)

    vocab_size = 100
    d_model = 128
    d_state = 64
    n_layers = 2
    headdim = 64
    batch_size = 8
    seq_len = 32

    print(f"📊 Configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  d_state: {d_state}")
    print(f"  headdim: {headdim}")
    print(f"  Layers: {n_layers}")
    task_desc = {
        "shift": "labels[i] = input[i-1]",
        "copy": "labels[i] = input[i]"
    }
    print(f"  Task: {task} ({task_desc.get(task, 'unknown')})")

    model = SimpleMamba2LM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        headdim=headdim,
        device=DEVICE
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔄 Training for {num_steps} steps...")
    initial_loss = None
    for step in range(1, num_steps + 1):
        input_ids, labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
        loss = train_step(model, input_ids, labels, optimizer, criterion)
        
        if step == 1:
            initial_loss = loss
        
        if step % 5 == 0 or step == 1:
            test_input, test_labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
            test_loss, accuracy = evaluate(model, test_input, test_labels, criterion)
            print(f"  Step {step:2d}/{num_steps}: Loss = {loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.2%}")

    if initial_loss is not None:
        final_test_input, final_test_labels = create_learnable_data(batch_size, seq_len, vocab_size, task)
        final_loss, final_accuracy = evaluate(model, final_test_input, final_test_labels, criterion)
        improvement = initial_loss - final_loss
        print(f"\n📈 Learning Progress:")
        print(f"  Initial Loss: {initial_loss:.4f}")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/initial_loss*100:.1f}% reduction)")
        print(f"  Final Accuracy: {final_accuracy:.2%}")

    print("✅ Mamba 2 training completed!")


def main():
    print("🎓 Simple Training Example")
    print("=" * 50)

    print(f"Device: {DEVICE}")

    task = "shift"
    num_steps = 200

    print(f"Task: {task}")
    print(f"Training steps: {num_steps}")

    train_mamba1(task, num_steps)
    train_mamba2(task, num_steps)

    print("\n📊 Summary")
    print("=" * 50)
    print("✅ Mamba 1: SSM architecture, d_state=16")
    print("✅ Mamba 2: SSD architecture, d_state=64, multi-head")
    print("\n💡 Both architectures support full training with backpropagation!")

if __name__ == "__main__":
    main()
