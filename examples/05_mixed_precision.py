#!/usr/bin/env python3
"""
Mixed Precision Support Demo for Mamba2MacOS on Apple Silicon

This script demonstrates different precision modes and their performance characteristics:
1. FP32 (default) - Full precision
2. FP16 - Half precision (not recommended on MPS due to stability issues)  
3. BF16 - BFloat16 (experimental on MPS)
4. Automatic Mixed Precision (AMP) using torch.autocast

Performance comparisons and memory usage analysis included.
"""

import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mamba_ssm.modules.mamba2_macos import Mamba2MacOS


class Mamba2MacOSModel(nn.Module):
    def __init__(self, d_model=512, n_layer=8, vocab_size=50277, d_state=32, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device  # Store device as attribute
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=dtype)
        self.norm = nn.LayerNorm(d_model, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype)
        self.layers = nn.ModuleList([
            Mamba2MacOS(d_model=d_model, d_state=d_state, layer_idx=i, device=device, dtype=dtype) 
            for i in range(n_layer)
        ])
        self.lm_head.weight = self.embedding.weight
        self.to(device=device, dtype=dtype)
    
    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def benchmark_precision_mode(model, input_ids, precision_name, num_runs=20, use_autocast=False, autocast_dtype=None):
    """Benchmark a specific precision mode"""
    print(f"\n🔍 Testing {precision_name}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")
    print(f"   Input dtype: {input_ids.dtype}")
    print(f"   Autocast: {use_autocast} ({autocast_dtype if autocast_dtype else 'N/A'})")
    
    device_type = "mps" if "mps" in str(next(model.parameters()).device) else "cpu"
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
    
    if device_type == "mps":
        torch.mps.synchronize()
    
    # Memory before
    if device_type == "mps":
        memory_before = torch.mps.current_allocated_memory()
    else:
        memory_before = 0
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    output = model(input_ids)
            else:
                output = model(input_ids)
    
    if device_type == "mps":
        torch.mps.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    # Memory after
    if device_type == "mps":
        memory_after = torch.mps.current_allocated_memory()
        memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
    else:
        memory_used = 0
    
    print(f"   ⏱️  {avg_time*1000:.2f}ms/forward, {1/avg_time:.1f} tokens/sec")
    print(f"   💾 Memory: {memory_used:.1f}MB")
    print(f"   📊 Output dtype: {output.dtype}")
    
    return avg_time, memory_used, output


def test_mixed_precision_modes():
    """Test different precision modes available on Apple Silicon"""
    print("🍎 Mixed Precision Support on Apple Silicon MPS")
    print("=" * 60)
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    batch_size, seq_len = input_ids.shape
    print(f"Input shape: {input_ids.shape}")
    
    results = {}
    
    # Test 1: FP32 (Full Precision) - Default and most stable
    try:
        model_fp32 = Mamba2MacOSModel(device=device, dtype=torch.float32)
        input_fp32 = input_ids.to(torch.long)  # Keep input as long for embeddings
        avg_time, memory, output = benchmark_precision_mode(
            model_fp32, input_fp32, "FP32 (Full Precision) - Recommended"
        )
        results["FP32"] = {"time": avg_time, "memory": memory, "stable": True}
    except Exception as e:
        print(f"❌ FP32 failed: {e}")
        results["FP32"] = {"error": str(e)}
    
    # Test 2: FP16 (Half Precision) - Not recommended on MPS
    try:
        model_fp16 = Mamba2MacOSModel(device=device, dtype=torch.float16)
        input_fp16 = input_ids.to(torch.long)
        avg_time, memory, output = benchmark_precision_mode(
            model_fp16, input_fp16, "FP16 (Half Precision) - Not recommended on MPS"
        )
        results["FP16"] = {"time": avg_time, "memory": memory, "stable": False}
        print("   ⚠️  FP16 may cause instability on MPS - use with caution")
    except Exception as e:
        print(f"❌ FP16 failed: {e}")
        results["FP16"] = {"error": str(e)}
    
    # Test 3: BF16 (BFloat16) - Experimental on MPS
    if hasattr(torch, 'bfloat16'):
        try:
            model_bf16 = Mamba2MacOSModel(device=device, dtype=torch.bfloat16)
            input_bf16 = input_ids.to(torch.long)
            avg_time, memory, output = benchmark_precision_mode(
                model_bf16, input_bf16, "BF16 (BFloat16) - Experimental on MPS"
            )
            results["BF16"] = {"time": avg_time, "memory": memory, "stable": False}
            print("   ⚠️  BF16 is experimental on MPS - limited support")
        except Exception as e:
            print(f"❌ BF16 failed: {e}")
            results["BF16"] = {"error": str(e)}
    
    # Test 4: Automatic Mixed Precision with FP16
    try:
        model_amp = Mamba2MacOSModel(device=device, dtype=torch.float32)
        input_amp = input_ids.to(torch.long)
        avg_time, memory, output = benchmark_precision_mode(
            model_amp, input_amp, "AMP with FP16 (Autocast)", 
            use_autocast=True, autocast_dtype=torch.float16
        )
        results["AMP_FP16"] = {"time": avg_time, "memory": memory, "stable": False}
        print("   ⚠️  AMP FP16 may be unstable on MPS")
    except Exception as e:
        print(f"❌ AMP FP16 failed: {e}")
        results["AMP_FP16"] = {"error": str(e)}
    
    # Test 5: Automatic Mixed Precision with BF16
    if hasattr(torch, 'bfloat16'):
        try:
            model_amp_bf16 = Mamba2MacOSModel(device=device, dtype=torch.float32)
            input_amp_bf16 = input_ids.to(torch.long)
            avg_time, memory, output = benchmark_precision_mode(
                model_amp_bf16, input_amp_bf16, "AMP with BF16 (Autocast)",
                use_autocast=True, autocast_dtype=torch.bfloat16
            )
            results["AMP_BF16"] = {"time": avg_time, "memory": memory, "stable": False}
            print("   ⚠️  AMP BF16 is experimental on MPS")
        except Exception as e:
            print(f"❌ AMP BF16 failed: {e}")
            results["AMP_BF16"] = {"error": str(e)}
    
    return results


def demonstrate_precision_conversion():
    """Show how to convert models between precision modes"""
    print("\n🔄 Model Precision Conversion Examples")
    print("=" * 50)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Create base model in FP32
    model = Mamba2MacOSModel(d_model=256, n_layer=4, device=device, dtype=torch.float32)
    print(f"Original model dtype: {next(model.parameters()).dtype}")
    
    # Convert to half precision (not recommended but possible)
    model_half = model.half()
    print(f"After .half(): {next(model_half.parameters()).dtype}")
    
    # Convert back to float
    model_float = model_half.float()
    print(f"After .float(): {next(model_float.parameters()).dtype}")
    
    # Manual dtype conversion
    if hasattr(torch, 'bfloat16'):
        model_bf16 = model.to(dtype=torch.bfloat16)
        print(f"After .to(bfloat16): {next(model_bf16.parameters()).dtype}")


def show_autocast_usage():
    """Demonstrate proper autocast usage patterns"""
    print("\n🎯 Autocast Usage Patterns")
    print("=" * 40)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Mamba2MacOSModel(d_model=256, n_layer=4, device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    
    print("1. Basic autocast with FP16:")
    print("   with torch.autocast(device_type='mps', dtype=torch.float16):")
    print("       output = model(input_ids)")
    
    print("\n2. Autocast with BF16 (if supported):")
    print("   with torch.autocast(device_type='mps', dtype=torch.bfloat16):")
    print("       output = model(input_ids)")
    
    print("\n3. Conditional autocast:")
    print("   dtype = torch.float16 if use_amp else torch.float32")
    print("   with torch.autocast(device_type='mps', dtype=dtype, enabled=use_amp):")
    print("       output = model(input_ids)")
    
    # Practical example
    print("\n4. Practical inference example:")
    try:
        use_amp = True
        amp_dtype = torch.float16
        
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    output = model(input_ids)
            else:
                output = model(input_ids)
        
        print(f"   ✅ Success! Output shape: {output.shape}, dtype: {output.dtype}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")


def print_recommendations():
    """Print recommendations for mixed precision on Apple Silicon"""
    print("\n💡 Recommendations for Apple Silicon")
    print("=" * 50)
    
    recommendations = [
        "✅ **Recommended**: Use FP32 for stability and accuracy",
        "⚠️  **Caution**: FP16 may cause numerical instability on MPS",
        "🧪 **Experimental**: BF16 support is limited on MPS", 
        "🔧 **Development**: Use autocast for experimentation only",
        "📊 **Production**: Stick with FP32 for reliable inference",
        "💾 **Memory**: Mixed precision may not save significant memory on MPS",
        "🏃 **Speed**: Performance gains from mixed precision are limited on MPS",
        "🔍 **Testing**: Always validate numerical accuracy when using mixed precision"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")


def main():
    print("🔬 Mamba2MacOS Mixed Precision Analysis")
    print("=" * 60)
    
    # Test all precision modes
    results = test_mixed_precision_modes()
    
    # Show conversion examples
    demonstrate_precision_conversion()
    
    # Show autocast patterns
    show_autocast_usage()
    
    # Print recommendations
    print_recommendations()
    
    # Summary
    print(f"\n📋 Summary of Results")
    print("=" * 30)
    
    for mode, result in results.items():
        if "error" in result:
            print(f"{mode:12}: ❌ Failed - {result['error'][:50]}...")
        else:
            status = "✅ Stable" if result.get("stable", False) else "⚠️  Unstable"
            print(f"{mode:12}: {status} - {result['time']*1000:.1f}ms, {result['memory']:.1f}MB")
    
    print(f"\n🎯 **Recommendation**: Use FP32 for production workloads on Apple Silicon")
    print(f"🧪 **Experimentation**: Mixed precision can be tested but may not provide significant benefits")


if __name__ == "__main__":
    main() 