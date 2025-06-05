#!/usr/bin/env python3
"""
Perfect Structure Demo - Showcasing the new organized codebase

This example demonstrates how to use the reorganized Mamba SSM macOS codebase
with clean imports and the new directory structure.

Features:
- Uses the new src/mamba_macos package structure
- Demonstrates both Mamba1 and Mamba2 models
- Shows proper error handling and device selection
- Interactive and automated modes
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for clean imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mamba_macos import (generate_text_with_model, get_device,
                         load_and_prepare_model)


def demo_model(model_name, prompts, model_dir="./models"):
    """Demonstrate a specific model with various prompts."""
    print(f"\n🤖 Testing {model_name.upper()}")
    print("=" * 50)
    
    device = get_device()
    print(f"📱 Device: {device}")
    
    # Load model
    success, model, tokenizer = load_and_prepare_model(model_name, model_dir, device)
    
    if not success:
        print(f"❌ Failed to load {model_name}")
        return False
    
    # Test with different prompts
    for i, prompt_config in enumerate(prompts, 1):
        prompt = prompt_config["prompt"]
        max_length = prompt_config.get("max_length", 40)
        temperature = prompt_config.get("temperature", 0.7)
        
        print(f"\n📝 Test {i}: '{prompt}'")
        print(f"⚙️ Settings: max_length={max_length}, temperature={temperature}")
        
        start_time = time.time()
        try:
            generated = generate_text_with_model(
                model, tokenizer, prompt, device, max_length, temperature
            )
            gen_time = time.time() - start_time
            
            print(f"✅ Generated ({gen_time:.2f}s):")
            print(f"   {generated}")
            
            # Calculate words per second
            words_generated = len(generated.split()) - len(prompt.split())
            if words_generated > 0 and gen_time > 0:
                wps = words_generated / gen_time
                print(f"📊 Performance: {wps:.1f} words/sec")
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    return True


def interactive_mode():
    """Interactive mode for custom prompts."""
    print("\n🎯 Interactive Mode")
    print("=" * 50)
    print("Enter prompts to test with both models (type 'quit' to exit)")
    
    device = get_device()
    models = {}
    
    # Pre-load both models
    for model_name in ["mamba1", "mamba2"]:
        print(f"\n🔄 Loading {model_name}...")
        success, model, tokenizer = load_and_prepare_model(model_name, "./models", device)
        if success:
            models[model_name] = (model, tokenizer)
            print(f"✅ {model_name} ready")
        else:
            print(f"❌ {model_name} failed to load")
    
    if not models:
        print("❌ No models available for interactive mode")
        return
    
    while True:
        try:
            prompt = input("\n💭 Enter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Get parameters
            try:
                max_length = int(input("📏 Max length (default 30): ") or "30")
                temperature = float(input("🌡️ Temperature (default 0.7): ") or "0.7")
            except ValueError:
                max_length, temperature = 30, 0.7
            
            # Test with all available models
            for model_name, (model, tokenizer) in models.items():
                print(f"\n🤖 {model_name.upper()}:")
                try:
                    generated = generate_text_with_model(
                        model, tokenizer, prompt, device, max_length, temperature
                    )
                    print(f"   {generated}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break


def showcase_structure():
    """Show the perfect directory structure."""
    print("\n🏗️ Perfect Directory Structure")
    print("=" * 50)
    
    structure_info = [
        ("📦 src/mamba_macos/", "Core library with clean imports"),
        ("🔧 scripts/", "Utility scripts (download, run models)"),
        ("🧪 tests/unit/", "Component-level unit tests"),
        ("🧪 tests/integration/", "End-to-end integration tests"),
        ("📚 examples/", "Usage examples and demos"),
        ("⚙️ config/", "Configuration files"),
        ("🛠️ tools/", "Development tools and test runners"),
        ("🤖 models/", "Downloaded model files"),
    ]
    
    for path, description in structure_info:
        print(f"{path:<25} {description}")
    
    print(f"\n🎯 Key Benefits:")
    print("✅ Clean separation of concerns")
    print("✅ Professional Python package structure") 
    print("✅ Easy imports: from mamba_macos import ...")
    print("✅ Organized tests (unit + integration)")
    print("✅ Makefile for common development tasks")


def main():
    parser = argparse.ArgumentParser(description="Perfect Structure Demo")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--model", choices=["mamba1", "mamba2", "both"], default="both", help="Model to test")
    parser.add_argument("--show-structure", action="store_true", help="Show directory structure")
    
    args = parser.parse_args()
    
    print("🎉 Perfect Structure Demo")
    print("=" * 50)
    print("Showcasing the reorganized Mamba SSM macOS codebase")
    
    if args.show_structure:
        showcase_structure()
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    # Predefined demo prompts
    demo_prompts = [
        {
            "prompt": "The future of artificial intelligence",
            "max_length": 35,
            "temperature": 0.7
        },
        {
            "prompt": "Apple Silicon processors are revolutionizing",
            "max_length": 30,
            "temperature": 0.8
        },
        {
            "prompt": "State space models represent",
            "max_length": 25,
            "temperature": 0.6
        }
    ]
    
    models_to_test = []
    if args.model in ["mamba1", "both"]:
        models_to_test.append("mamba1")
    if args.model in ["mamba2", "both"]:
        models_to_test.append("mamba2")
    
    success_count = 0
    for model_name in models_to_test:
        if demo_model(model_name, demo_prompts):
            success_count += 1
    
    print(f"\n🎯 Demo Summary")
    print("=" * 50)
    print(f"✅ Successfully tested {success_count}/{len(models_to_test)} models")
    print("📋 Try these commands:")
    print("  python -m examples.09_perfect_structure_demo --interactive")
    print("  python -m examples.09_perfect_structure_demo --show-structure")
    print("  python -m examples.09_perfect_structure_demo --model mamba1")
    print("\n🛠️ Development commands:")
    print("  make test-quick    # Quick integration test")
    print("  make run-mamba1    # Run Mamba1 demo")
    print("  make show-structure # Show project layout")


if __name__ == "__main__":
    main() 