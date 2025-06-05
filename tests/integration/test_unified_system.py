#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

# Add src and scripts to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

from download_models import download_model
from mamba_macos import (generate_text_with_model, get_device,
                         load_and_prepare_model)


def test_device_selection():
    device = get_device()
    print(f"📱 Selected device: {device}")
    
    if device == "mps":
        assert torch.backends.mps.is_available() and torch.backends.mps.is_built()
        print("✅ MPS acceleration available")
    elif device == "cuda":
        assert torch.cuda.is_available()
        print("✅ CUDA acceleration available")
    else:
        print("✅ Using CPU (no acceleration)")
    
    return device

def test_download_functionality(temp_dir):
    print("\n🔽 Testing download functionality...")
    
    for model_type in ["mamba1", "mamba2"]:
        print(f"📦 Testing {model_type} download...")
        try:
            download_model(model_type, temp_dir)
            
            expected_files = [f"{model_type}-130m-config.json", f"{model_type}-130m-model.bin"]
            model_dir = Path(temp_dir) / model_type
            
            for file_name in expected_files:
                file_path = model_dir / file_name
                assert file_path.exists(), f"{file_name} not found"
                print(f"   ✅ {file_name} downloaded")
            
            config_path = model_dir / f"{model_type}-130m-config.json"
            with open(config_path) as f:
                config = json.load(f)
            assert "d_model" in config and "ssm_cfg" in config
            print(f"   ✅ {model_type} config validated")
            
        except Exception as e:
            print(f"   ❌ {model_type} download failed: {e}")
            return False
    
    print("✅ All downloads successful")
    return True

def test_model_loading(model_dir, device):
    print("\n🏗️ Testing model loading...")
    
    for model_type in ["mamba1", "mamba2"]:
        print(f"🔄 Loading {model_type}...")
        try:
            success, model, tokenizer = load_and_prepare_model(model_type, model_dir, device)
            
            if not success:
                print(f"   ❌ {model_type} loading failed")
                return False
            
            assert model is not None and tokenizer is not None and hasattr(model, 'forward')
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ {model_type} loaded: {total_params:,} parameters")
            
        except Exception as e:
            print(f"   ❌ {model_type} loading error: {e}")
            return False
    
    print("✅ All models loaded successfully")
    return True

def test_text_generation(model_dir, device):
    print("\n📝 Testing text generation...")
    
    test_prompts = ["The weather today is", "AI technology", "Machine learning"]
    
    for model_type in ["mamba1", "mamba2"]:
        print(f"🤖 Testing {model_type} generation...")
        try:
            success, model, tokenizer = load_and_prepare_model(model_type, model_dir, device)
            if not success:
                print(f"   ❌ Failed to load {model_type}")
                return False
            
            for prompt in test_prompts:
                generated = generate_text_with_model(model, tokenizer, prompt, device, max_length=30, temperature=0.7)
                
                assert generated is not None and len(generated) > len(prompt) and isinstance(generated, str)
                print(f"   ✅ '{prompt}' → Generated {len(generated)} chars")
            
        except Exception as e:
            print(f"   ❌ {model_type} generation error: {e}")
            return False
    
    print("✅ All generation tests passed")
    return True

def test_parameter_variations(model_dir, device):
    print("\n🎛️ Testing parameter variations...")
    
    success, model, tokenizer = load_and_prepare_model("mamba1", model_dir, device)
    if not success:
        print("❌ Failed to load model for parameter testing")
        return False
    
    prompt = "Test generation"
    
    try:
        short = generate_text_with_model(model, tokenizer, prompt, device, max_length=20, temperature=0.5)
        long = generate_text_with_model(model, tokenizer, prompt, device, max_length=50, temperature=0.5)
        cold = generate_text_with_model(model, tokenizer, prompt, device, max_length=30, temperature=0.1)
        hot = generate_text_with_model(model, tokenizer, prompt, device, max_length=30, temperature=1.0)
        
        assert len(short) <= len(long) and all(isinstance(text, str) for text in [short, long, cold, hot])
        print("✅ Length and temperature parameters working")
        
    except Exception as e:
        print(f"❌ Parameter testing error: {e}")
        return False
    
    print("✅ Parameter variation tests passed")
    return True

def run_comprehensive_test():
    print("🧪 Starting comprehensive unified system test...")
    
    device = test_device_selection()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        tests = [
            (test_download_functionality, (temp_dir,), "Download tests"),
            (test_model_loading, (temp_dir, device), "Model loading tests"),
            (test_text_generation, (temp_dir, device), "Text generation tests"),
            (test_parameter_variations, (temp_dir, device), "Parameter variation tests")
        ]
        
        for test_func, args, test_name in tests:
            if not test_func(*args):
                print(f"❌ {test_name} failed")
                return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Unified system is working perfectly")
    return True

def run_quick_test(model_dir):
    print("⚡ Running quick test with existing models...")
    
    device = get_device()
    
    for model_type in ["mamba1", "mamba2"]:
        print(f"\n🤖 Testing {model_type}...")
        success, model, tokenizer = load_and_prepare_model(model_type, model_dir, device)
        
        if success:
            generated = generate_text_with_model(model, tokenizer, "Hello world", device, max_length=20, temperature=0.7)
            print(f"✅ {model_type}: '{generated[:50]}...'")
        else:
            print(f"❌ {model_type} not available")
    
    print("⚡ Quick test completed")

def main():
    parser = argparse.ArgumentParser(description="Test unified Mamba system")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test with downloads")
    parser.add_argument("--model-dir", default="./models", help="Existing model directory for quick test")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        success = run_comprehensive_test()
        print("✅ SUCCESS" if success else "❌ FAILED")
    else:
        run_quick_test(args.model_dir)

if __name__ == "__main__":
    main() 