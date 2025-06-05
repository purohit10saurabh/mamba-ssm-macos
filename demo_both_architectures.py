import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig


def test_architecture(arch_name, config_data, test_weights=False):
    print(f"\n🧪 Testing {arch_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = MambaConfig(**config_data)
    
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
        
        total_params = sum(p.numel() for p in model.parameters())
        mixer_class = model.backbone.layers[0].mixer.__class__.__name__
        print(f"✅ {arch_name}: {total_params:,} params, {mixer_class}")
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        test_prompt = f"{arch_name} is working"
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            print(f"✅ Forward pass: {outputs.logits.shape}")
        
        if test_weights:
            model_file = Path(f"models/{arch_name.lower()}-130m-model.bin")
            if model_file.exists():
                try:
                    state_dict = torch.load(model_file, map_location="cpu")
                    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
                    print(f"📦 Weights: {len(missing_keys)} missing keys")
                except Exception:
                    print("⚠️ Weight loading failed (expected for mismatched architectures)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 Mamba Architecture Comparison")
    
    mamba1_config = {"d_model": 768, "n_layer": 4, "vocab_size": 50280, "ssm_cfg": {"layer": "Mamba1"}, "fused_add_norm": False}
    mamba2_config = {"d_model": 768, "n_layer": 4, "vocab_size": 50280, "ssm_cfg": {"layer": "Mamba2", "d_state": 64, "headdim": 128, "expand": 2}, "fused_add_norm": False, "rms_norm": True}
    
    mamba1_success = test_architecture("Mamba1", mamba1_config)
    mamba2_success = test_architecture("Mamba2", mamba2_config)
    
    print(f"\n🎯 Results: Mamba1 {'✅' if mamba1_success else '❌'}, Mamba2 {'✅' if mamba2_success else '❌'}")
    
    if mamba1_success and mamba2_success:
        print("🎉 Both architectures supported!")

if __name__ == "__main__":
    main() 