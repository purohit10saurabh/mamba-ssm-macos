#!/usr/bin/env python3
"""
Example script for running text generation with Mamba on macOS Apple Silicon.
This provides a simple interface for generating text using the Mamba model.
"""

import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mamba_ssm.modules.mamba_simple import Mamba


class SimpleMambaLMHeadModel(nn.Module):
    """A simple language model using Mamba for text generation on macOS."""
    
    def __init__(self, d_model=256, n_layer=4, vocab_size=50277, d_state=16, 
                 device=None, dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create a stack of Mamba layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            ) for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Move model to specified device and dtype
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.to(device=device, dtype=dtype)
        self.device = device
        
    def forward(self, input_ids):
        """Forward pass for generation."""
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=0.7, top_p=0.9, 
                repetition_penalty=1.2, stop_token=None):
        """Generate text using the model.
        
        Args:
            input_ids: Input token ids
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_token: Token id to stop generation at (optional)
        """
        generated = input_ids.clone()
        past_tokens = set()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            with torch.no_grad():
                logits = self.forward(generated)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for token in past_tokens:
                next_token_logits[:, token] /= repetition_penalty
            
            # Apply top-p (nucleus) sampling
            probs = torch.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update past tokens
            past_tokens.add(next_token.item())
            
            # Check for stop token
            if stop_token is not None and next_token.item() == stop_token:
                break
        
        return generated


def main():
    parser = argparse.ArgumentParser(description="Run text generation with Mamba on macOS")
    parser.add_argument("--model-size", type=str, default="small", 
                      choices=["small", "medium", "large"],
                      help="Model size to use")
    parser.add_argument("--prompt", type=str, required=True,
                      help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                      help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                      help="Top-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                      help="Repetition penalty")
    args = parser.parse_args()
    
    # Model configuration based on size
    model_configs = {
        "small": {"d_model": 256, "n_layer": 4, "d_state": 16},
        "medium": {"d_model": 512, "n_layer": 8, "d_state": 32},
        "large": {"d_model": 1024, "n_layer": 12, "d_state": 64}
    }
    config = model_configs[args.model_size]
    
    # Initialize tokenizer (using GPT-2 tokenizer as an example)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Initialize model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SimpleMambaLMHeadModel(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        vocab_size=tokenizer.vocab_size,
        d_state=config["d_state"],
        device=device
    )
    
    # Tokenize input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    # Generate
    print(f"\nGenerating with prompt: {args.prompt}\n")
    output_ids = model.generate(
        input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token=tokenizer.eos_token_id
    )
    
    # Decode and print output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text:\n{output_text}")


if __name__ == "__main__":
    main() 