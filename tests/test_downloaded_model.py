#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import pytest
import torch

from run_mamba import generate_text, load_downloaded_model


@pytest.fixture
def mock_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        
        config_data = {
            "d_model": 768,
            "n_layer": 24,
            "vocab_size": 50280,
            "ssm_cfg": {"layer": "Mamba1"},
            "fused_add_norm": False
        }
        
        with open(model_dir / "mamba-130m-config.json", "w") as f:
            json.dump(config_data, f)
        
        dummy_weights = {"dummy": torch.randn(10, 10)}
        torch.save(dummy_weights, model_dir / "mamba-130m-model.bin")
        
        yield model_dir

def test_config_loading(mock_model_dir):
    config_file = mock_model_dir / "mamba-130m-config.json"
    assert config_file.exists()
    
    with open(config_file) as f:
        config = json.load(f)
    
    assert config["d_model"] == 768
    assert config["n_layer"] == 24
    assert config["ssm_cfg"]["layer"] == "Mamba1"

@pytest.mark.skip(reason="Requires actual model files")
def test_model_loading():
    device = "cpu"
    model_path = "models"
    
    model, tokenizer = load_downloaded_model(model_path, device)
    
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, 'forward')
    assert hasattr(model, 'generate')

@pytest.mark.skip(reason="Requires actual model files")
def test_text_generation():
    device = "cpu"
    model_path = "models"
    
    model, tokenizer = load_downloaded_model(model_path, device)
    
    prompt = "The future of AI"
    generated_text, gen_time = generate_text(model, tokenizer, prompt, max_length=20, temperature=0.7, device=device)
    
    assert generated_text is not None
    assert len(generated_text) > len(prompt)
    assert gen_time > 0
    assert isinstance(generated_text, str)

@pytest.mark.skip(reason="Requires actual model files")
def test_generation_parameters():
    device = "cpu"
    model_path = "models"
    
    model, tokenizer = load_downloaded_model(model_path, device)
    
    prompt = "Test prompt"
    
    short_text, short_time = generate_text(model, tokenizer, prompt, max_length=10, temperature=0.5, device=device)
    long_text, long_time = generate_text(model, tokenizer, prompt, max_length=30, temperature=0.5, device=device)
    
    assert len(short_text) < len(long_text)
    assert short_time < long_time

def test_device_availability():
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    if mps_available:
        assert torch.backends.mps.is_built()
    
    assert isinstance(mps_available, bool)
    assert isinstance(cuda_available, bool)

def test_missing_files_handling(mock_model_dir):
    (mock_model_dir / "mamba-130m-config.json").unlink()
    
    model, tokenizer = load_downloaded_model(str(mock_model_dir), "cpu")
    assert model is None
    assert tokenizer is None

if __name__ == "__main__":
    print("🧪 Running basic tests...")
    
    print("✅ Testing device availability...")
    test_device_availability()
    print("✅ Device tests passed")
    
    print("✅ Testing file handling...")
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_dir = Path(temp_dir)
        config_data = {"d_model": 768, "n_layer": 24, "vocab_size": 50280, "ssm_cfg": {"layer": "Mamba1"}, "fused_add_norm": False}
        with open(mock_dir / "mamba-130m-config.json", "w") as f:
            json.dump(config_data, f)
        torch.save({"dummy": torch.randn(5, 5)}, mock_dir / "mamba-130m-model.bin")
        test_config_loading(mock_dir)
        test_missing_files_handling(mock_dir)
    print("✅ File handling tests passed")
    
    print("\n🎯 All basic tests completed successfully!")
    print("📝 Note: Full model tests require actual model files")
    print("   Run with pytest for complete test suite") 