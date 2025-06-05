#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def test_mamba2():
    print("Testing Mamba2-130m Model")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Try official config first, fallback to custom config
    config_files = ["models/mamba2_config.json", "models/official_mamba2/config.json", "models/mamba2_final_config.json"]
    config = None
    for config_file in config_files:
        if Path(config_file).exists():
            config = MambaConfig(**json.load(open(config_file)))
            print(f"Using config: {config_file}")
            break
    if not config:
        print("❌ No config file found")
        return False
    
    try:
        model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
        state_dict = torch.load("models/official_mamba2/pytorch_model.bin", map_location="cpu")
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params, {len(missing_keys)} missing keys")
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
        
        test_prompts = ["The future of artificial intelligence is", "Once upon a time", 
                       "Python is a programming language that", "The capital of France is"]
        
        print("\n🎯 Generation Examples:")
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = input_ids.clone()
                for _ in range(8):
                    outputs = model(generated)
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                print(f"  📝 '{prompt}' -> '{tokenizer.decode(generated[0], skip_special_tokens=True)}'")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("✅ SUCCESS" if test_mamba2() else "❌ FAILED") 