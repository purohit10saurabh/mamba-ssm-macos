#!/usr/bin/env python3
"""
Mamba2MacOS Example Script
--------------------------

This is the main example script for running Mamba2 on macOS. It provides a comprehensive
interface for text generation and benchmarking using the Mamba2MacOS implementation.

For specialized demos and examples, see the following files in the examples/macos/ directory:
- selective_scan_demo.py: Demonstrates the selective scan operation
- minimal_ssm_demo.py: Shows a minimal SSM implementation
- basic_demo.py: Basic Mamba functionality demo
- benchmark.py: Detailed benchmarking tools

Usage:
    # Basic text generation
    python run_mamba2_macos_example.py --prompt "Once upon a time"

    # Run benchmark
    python run_mamba2_macos_example.py --benchmark --model-size large

    # Generate with custom parameters
    python run_mamba2_macos_example.py \
        --model-size large \
        --prompt "Once upon a time" \
        --max-length 200 \
        --temperature 0.7 \
        --top-k 40 \
        --top-p 0.95
"""

import argparse
import time
from typing import List, Optional, Set

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS
from mamba_ssm.utils.generation import InferenceParams


class Mamba2MacOSModel(nn.Module):
    """A simple model using Mamba2MacOS for sequence modeling."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 4,
        vocab_size: int = 50277,
        d_state: int = 16,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create a stack of Mamba2MacOS layers
        self.layers = nn.ModuleList([
            Mamba2MacOS(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                layer_idx=i,  # Required for inference cache
                device=device,
                dtype=dtype
            ) for i in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Move model to specified device and dtype
        self.to(device=device, dtype=dtype)
        self.device = device
    
    def forward(self, input_ids: torch.Tensor, inference_params: Optional[InferenceParams] = None):
        """Forward pass through the model."""
        hidden_states = self.embedding(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token: Optional[int] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            input_ids: Input token ids
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_token: Token id to stop generation at (optional)
            verbose: Whether to print generation progress
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = input_ids.dtype
        
        # Initialize inference parameters
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        
        # Allocate cache for each layer
        for layer in self.layers:
            layer.allocate_inference_cache(batch_size, max_length, dtype=dtype)
        
        # Track generated tokens for repetition penalty
        past_tokens: Set[int] = set()
        generated = input_ids.clone()
        
        # Generate tokens
        for i in range(max_length - input_ids.shape[1]):
            if verbose and i % 10 == 0:
                print(f"Generating token {i+1}/{max_length - input_ids.shape[1]}...", end="\r")
            
            # Get logits for next token
            with torch.no_grad():
                logits = self(input_ids, inference_params=inference_params)
                logits = logits[:, -1]  # Get last token logits
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in past_tokens:
                    logits[:, token] /= repetition_penalty
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update past tokens
            past_tokens.add(next_token.item())
            
            # Check for stop token
            if stop_token is not None and next_token.item() == stop_token:
                if verbose:
                    print(f"\nStopping generation at token {i+1} due to stop token")
                break
        
        if verbose:
            print("\nGeneration complete!")
        
        return generated


def benchmark_model(model: Mamba2MacOSModel, input_ids: torch.Tensor, n_runs: int = 10) -> float:
    """Benchmark model generation speed."""
    print("\nRunning benchmark...")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Benchmark
    if model.device == "mps":
        torch.mps.synchronize()
    
    start_time = time.time()
    
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_ids)
    
    if model.device == "mps":
        torch.mps.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs
    
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Mamba2MacOS Example")
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium", "large"],
                      help="Model size to use")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                      help="Device to run on (cpu or mps)")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model",
                      help="Prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                      help="Maximum length of generated sequence")
    parser.add_argument("--temperature", type=float, default=0.8,
                      help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                      help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                      help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                      help="Repetition penalty")
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmark instead of generation")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                      help="Number of runs for benchmark")
    args = parser.parse_args()
    
    # Model size configurations
    model_configs = {
        "small": {"d_model": 256, "n_layer": 4, "d_state": 16},
        "medium": {"d_model": 512, "n_layer": 8, "d_state": 32},
        "large": {"d_model": 1024, "n_layer": 12, "d_state": 64},
    }
    
    # Create model
    config = model_configs[args.model_size]
    print(f"\nCreating {args.model_size} model with:")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_layer: {config['n_layer']}")
    print(f"  d_state: {config['d_state']}")
    print(f"  device: {args.device}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create model
    model = Mamba2MacOSModel(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        vocab_size=tokenizer.vocab_size,
        d_state=config["d_state"],
        device=args.device
    )
    
    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    print(f"\nInput prompt: {args.prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    if args.benchmark:
        # Run benchmark
        avg_time = benchmark_model(model, input_ids, n_runs=args.benchmark_runs)
        print(f"\nBenchmark results:")
        print(f"  Average forward pass time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {1/avg_time:.1f} tokens/second")
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {args.device}")
    else:
        # Generate
        print("\nGenerating...")
        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token=tokenizer.eos_token_id,
            verbose=True
        )
        generation_time = time.time() - start_time
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text ({generation_time:.2f}s):")
        print(output_text)
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {args.device}")
        print(f"  Generation speed: {len(output_text.split()) / generation_time:.1f} tokens/second")


if __name__ == "__main__":
    main() 