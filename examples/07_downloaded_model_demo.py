#!/usr/bin/env python3
"""
Downloaded Mamba Model Demo - Showcase the working pre-trained model

This example demonstrates how to use the downloaded pre-trained Mamba model
for text generation on macOS Apple Silicon.

Usage:
    python examples/07_downloaded_model_demo.py
    python -m examples.07_downloaded_model_demo --prompt "Your prompt here"
    python -m examples.07_downloaded_model_demo --interactive
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import torch

from run_mamba import generate_text, load_downloaded_model


def interactive_mode(model, tokenizer, device):
    """Interactive text generation mode."""
    print("\n🎯 Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n💭 Enter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                print("⚠️ Please enter a non-empty prompt")
                continue
            
            # Get generation parameters
            try:
                max_length = int(input("📏 Max length (default 50): ") or "50")
                temperature = float(input("🌡️ Temperature (default 0.8): ") or "0.8")
            except ValueError:
                print("⚠️ Invalid input, using defaults")
                max_length, temperature = 50, 0.8
            
            print(f"\n🚀 Generating text...")
            generated_text, gen_time = generate_text(
                model, tokenizer, prompt, 
                max_length=max_length, 
                temperature=temperature, 
                device=device
            )
            
            if generated_text:
                print(f"\n📝 Generated text ({gen_time:.2f}s):")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
                
                # Calculate performance metrics
                words_generated = len(generated_text.split()) - len(prompt.split())
                if words_generated > 0 and gen_time > 0:
                    wps = words_generated / gen_time
                    print(f"📊 Performance: {wps:.1f} words/sec, {words_generated} words in {gen_time:.2f}s")
            else:
                print("❌ Generation failed")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_prompts(model, tokenizer, device):
    """Demonstrate with various interesting prompts."""
    demo_prompts = [
        {
            "prompt": "The future of artificial intelligence is",
            "max_length": 80,
            "temperature": 0.7,
            "description": "🤖 AI Future Prediction"
        },
        {
            "prompt": "Once upon a time, in a land far away,",
            "max_length": 100,
            "temperature": 0.9,
            "description": "📚 Creative Storytelling"
        },
        {
            "prompt": "The key to happiness is",
            "max_length": 60,
            "temperature": 0.6,
            "description": "💭 Philosophical Insight"
        },
        {
            "prompt": "In the year 2050,",
            "max_length": 70,
            "temperature": 0.8,
            "description": "🔮 Future Scenario"
        },
        {
            "prompt": "Scientists have discovered",
            "max_length": 75,
            "temperature": 0.7,
            "description": "🔬 Scientific Discovery"
        }
    ]
    
    print("\n🎭 Demo Mode - Showcasing Various Prompts")
    print("=" * 60)
    
    total_time, total_words = 0, 0
    
    for i, demo in enumerate(demo_prompts, 1):
        print(f"\n{demo['description']} ({i}/{len(demo_prompts)})")
        print(f"💭 Prompt: '{demo['prompt']}'")
        print(f"🔧 Settings: max_length={demo['max_length']}, temperature={demo['temperature']}")
        
        start_time = time.time()
        generated_text, gen_time = generate_text(
            model, tokenizer, demo['prompt'],
            max_length=demo['max_length'],
            temperature=demo['temperature'],
            device=device
        )
        
        if generated_text:
            print(f"\n📝 Result ({gen_time:.2f}s):")
            print("-" * 40)
            print(generated_text)
            print("-" * 40)
            
            words_generated = len(generated_text.split()) - len(demo['prompt'].split())
            total_words += words_generated
            total_time += gen_time
            
            if words_generated > 0 and gen_time > 0:
                wps = words_generated / gen_time
                print(f"📊 {wps:.1f} words/sec")
        else:
            print("❌ Generation failed")
        
        # Small pause between demos
        if i < len(demo_prompts):
            time.sleep(1)
    
    # Overall performance summary
    if total_time > 0 and total_words > 0:
        avg_wps = total_words / total_time
        print(f"\n📈 Overall Performance: {avg_wps:.1f} words/sec across {len(demo_prompts)} prompts")

def benchmark_mode(model, tokenizer, device):
    """Run performance benchmarks."""
    print("\n⚡ Benchmark Mode")
    print("=" * 60)
    
    benchmark_configs = [
        {"name": "Short Generation", "max_length": 20, "runs": 5},
        {"name": "Medium Generation", "max_length": 50, "runs": 3},
        {"name": "Long Generation", "max_length": 100, "runs": 2},
    ]
    
    prompt = "Performance test prompt for benchmarking"
    
    for config in benchmark_configs:
        print(f"\n🔬 {config['name']} (max_length={config['max_length']})")
        times, word_counts = [], []
        
        for run in range(config['runs']):
            print(f"   Run {run + 1}/{config['runs']}...", end=" ")
            
            generated_text, gen_time = generate_text(
                model, tokenizer, prompt,
                max_length=config['max_length'],
                temperature=0.7,
                device=device
            )
            
            if generated_text:
                times.append(gen_time)
                words = len(generated_text.split()) - len(prompt.split())
                word_counts.append(words)
                print(f"{gen_time:.2f}s")
            else:
                print("Failed")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_words = sum(word_counts) / len(word_counts)
            avg_wps = avg_words / avg_time if avg_time > 0 else 0
            
            print(f"   📊 Average: {avg_time:.2f}s, {avg_words:.1f} words, {avg_wps:.1f} words/sec")

def system_info():
    """Display system information."""
    print("🍎 Mamba Model Demo - System Information")
    print("=" * 60)
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 Device: {'MPS (Apple Silicon)' if torch.backends.mps.is_available() else 'CPU'}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS acceleration enabled")
    else:
        print("⚠️ MPS not available, using CPU")
    
    # Check model files
    model_dir = Path("models")
    config_file = model_dir / "mamba-130m-config.json"
    model_file = model_dir / "mamba-130m-model.bin"
    
    print(f"\n📁 Model Files:")
    print(f"   Config: {'✅' if config_file.exists() else '❌'} {config_file}")
    print(f"   Weights: {'✅' if model_file.exists() else '❌'} {model_file}")
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Downloaded Mamba Model Demo")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--demo", action="store_true", help="Demo mode with preset prompts")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark mode")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    
    args = parser.parse_args()
    
    # Display system info
    system_info()
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load model
    print(f"\n🔧 Loading model from {args.model_dir}...")
    model, tokenizer = load_downloaded_model(args.model_dir, device)
    
    if model is None:
        print("❌ Failed to load model. Please check:")
        print("   1. Model files exist in the specified directory")
        print("   2. Run the download script if needed")
        print("   3. Check the model directory path")
        return
    
    print("✅ Model loaded successfully!")
    
    # Determine mode
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.demo:
        demo_prompts(model, tokenizer, device)
    elif args.benchmark:
        benchmark_mode(model, tokenizer, device)
    elif args.prompt:
        # Single prompt mode
        print(f"\n💭 Generating for: '{args.prompt}'")
        generated_text, gen_time = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            device=device
        )
        
        if generated_text:
            print(f"\n📝 Generated text ({gen_time:.2f}s):")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            
            words_generated = len(generated_text.split()) - len(args.prompt.split())
            if words_generated > 0 and gen_time > 0:
                wps = words_generated / gen_time
                print(f"📊 Performance: {wps:.1f} words/sec")
        else:
            print("❌ Generation failed")
    else:
        # Default: show options
        print("\n🎯 Available modes:")
        print("   --prompt 'text'    Generate from single prompt")
        print("   --interactive      Interactive generation")
        print("   --demo            Demo with preset prompts")
        print("   --benchmark       Performance benchmarks")
        print("\nExample: python examples/07_downloaded_model_demo.py --demo")

if __name__ == "__main__":
    main() 