"""
Mamba2MacOS Example Script - Comprehensive interface for text generation and benchmarking.

Usage Examples:
    python run_mamba2_macos_example.py --prompt "Once upon a time"
    python run_mamba2_macos_example.py --benchmark --model-size large
    python run_mamba2_macos_example.py --model-size large --prompt "Once upon a time" --max-length 200 --temperature 0.7 --top-k 40 --top-p 0.95
"""

import argparse
import time
from typing import Optional, Set

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS
from mamba_ssm.utils.generation import InferenceParams


class Mamba2MacOSModel(nn.Module):
    def __init__(self, d_model: int = 256, n_layer: int = 4, vocab_size: int = 50277, 
                 d_state: int = 16, device: str = "cpu", dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.layers = nn.ModuleList([
            Mamba2MacOS(d_model=d_model, d_state=d_state, d_conv=4, expand=2, 
                       layer_idx=layer_idx, device=device, dtype=dtype) 
            for layer_idx in range(n_layer)
        ])
        
        self.lm_head.weight = self.embedding.weight
        self.device = device
        self.to(device=device, dtype=dtype)
    
    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
    
    def generate(self, input_ids, max_length: int, temperature: float = 1.0, 
                top_k: int = 1, top_p: float = 1.0, repetition_penalty: float = 1.0, 
                stop_token: Optional[int] = None, verbose: bool = True):
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        
        for layer_idx, layer in enumerate(self.layers):
            cache_conv, cache_ssm = layer.allocate_inference_cache(batch_size, max_length, dtype=input_ids.dtype)
            inference_params.key_value_memory_dict[layer_idx] = (cache_conv, cache_ssm)
        
        past_tokens = set()
        generated_sequence = input_ids.clone()
        
        tokens_to_generate = max_length - input_ids.shape[1]
        
        for token_idx in range(tokens_to_generate):
            if verbose and token_idx % 10 == 0:
                print(f"Generating token {token_idx+1}/{tokens_to_generate}...", end="\r")
            
            with torch.no_grad():
                last_token = generated_sequence[:, -1].unsqueeze(-1)
                logits = self(last_token, inference_params=inference_params)
            
            if temperature > 0:
                logits = logits / temperature
            
            if top_k > 0:
                topk_values, _ = torch.topk(logits, top_k)
                min_topk = topk_values[..., -1, None]
                logits[logits < min_topk] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                tokens_to_remove = cumulative_probs > top_p
                tokens_to_remove[..., 0] = False
                
                filtered_logits = torch.where(tokens_to_remove, float('-inf'), sorted_logits)
                logits.scatter_(-1, sorted_indices, filtered_logits)
            
            if repetition_penalty != 1.0 and past_tokens:
                past_token_tensor = torch.tensor(list(past_tokens), device=device)
                past_indices = past_token_tensor.unsqueeze(0).unsqueeze(0).expand(logits.shape[0], 1, -1)
                
                repeated_logits = torch.gather(logits, -1, past_indices)
                penalized_logits = torch.where(
                    repeated_logits >= 0, 
                    repeated_logits / repetition_penalty,
                    repeated_logits * repetition_penalty
                )
                logits.scatter_(-1, past_indices, penalized_logits)
            
            finite_logits_exist = torch.isfinite(logits).sum() > 0
            if not finite_logits_exist:
                if verbose:
                    print("\nWarning: All logits are -inf! Stopping generation.")
                break
            
            probabilities = torch.softmax(logits, dim=-1).squeeze(1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            past_tokens.add(next_token.item())
            
            if stop_token and next_token.item() == stop_token:
                if verbose:
                    print(f"\nStopping at token {token_idx+1} due to stop token")
                break
        
        if verbose:
            print("\nGeneration complete!")
        
        return generated_sequence


def benchmark_model_performance(model, input_ids, num_runs=10):
    print("\nRunning benchmark...")
    
    for _ in range(3):
        with torch.no_grad():
            model(input_ids)
    
    if model.device == "mps":
        torch.mps.synchronize()
    
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_ids)
    
    if model.device == "mps":
        torch.mps.synchronize()
    
    total_time = time.time() - start_time
    return total_time / num_runs


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Mamba2MacOS Example")
    
    argument_specs = [
        ("--model-size", {"type": str, "default": "small", "choices": ["small", "medium", "large"], "help": "Model size"}),
        ("--device", {"type": str, "default": "mps" if torch.backends.mps.is_available() else "cpu", "help": "Device"}),
        ("--prompt", {"type": str, "default": "Hello, I am a language model", "help": "Prompt"}),
        ("--max-length", {"type": int, "default": 100, "help": "Max length"}),
        ("--temperature", {"type": float, "default": 0.8, "help": "Temperature"}),
        ("--top-k", {"type": int, "default": 50, "help": "Top-k"}),
        ("--top-p", {"type": float, "default": 0.9, "help": "Top-p"}),
        ("--repetition-penalty", {"type": float, "default": 1.2, "help": "Repetition penalty"}),
        ("--benchmark", {"action": "store_true", "help": "Benchmark mode"}),
        ("--benchmark-runs", {"type": int, "default": 10, "help": "Benchmark runs"})
    ]
    
    for arg_name, arg_config in argument_specs:
        parser.add_argument(arg_name, **arg_config)
    
    return parser.parse_args()


def get_model_configuration(model_size: str):
    configurations = {
        "small": {"d_model": 256, "n_layer": 4, "d_state": 16},
        "medium": {"d_model": 512, "n_layer": 8, "d_state": 32},
        "large": {"d_model": 1024, "n_layer": 12, "d_state": 64}
    }
    return configurations[model_size]


def main():
    args = parse_command_line_arguments()
    
    model_config = get_model_configuration(args.model_size)
    print(f"\nCreating {args.model_size} model: d_model={model_config['d_model']}, "
          f"n_layer={model_config['n_layer']}, d_state={model_config['d_state']}, device={args.device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = Mamba2MacOSModel(**model_config, vocab_size=tokenizer.vocab_size, device=args.device)
    
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    print(f"Input: '{args.prompt}' (shape: {input_ids.shape})")
    
    if args.benchmark:
        average_time = benchmark_model_performance(model, input_ids, args.benchmark_runs)
        total_parameters = sum(param.numel() for param in model.parameters())
        
        print(f"Results: {average_time*1000:.2f}ms/forward, {1/average_time:.1f} tokens/sec, "
              f"{total_parameters:,} params, {args.device}")
    else:
        start_time = time.time()
        
        output_ids = model.generate(
            input_ids, args.max_length, args.temperature, args.top_k, 
            args.top_p, args.repetition_penalty, tokenizer.eos_token_id, True
        )
        
        generation_time = time.time() - start_time
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        total_parameters = sum(param.numel() for param in model.parameters())
        
        tokens_per_second = len(generated_text.split()) / generation_time
        
        print(f"\nGenerated ({generation_time:.2f}s):\n{generated_text}")
        print(f"\nStats: {total_parameters:,} params, {args.device}, {tokens_per_second:.1f} tokens/sec")


if __name__ == "__main__": 
    main() 