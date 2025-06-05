#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def main():
    parser = argparse.ArgumentParser(description="Mamba 2 Demo")
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    print("🆕 Mamba 2 Demo - SSD Model")
    device = "mps" if (args.device == "auto" and torch.backends.mps.is_available()) else args.device if args.device != "auto" else "cpu"
    print(f"🔧 Device: {device}")
    
    config_files = [Path("models/mamba2_config.json"), Path("models/official_mamba2/config.json")]
    model_file = Path("models/official_mamba2/pytorch_model.bin")
    config_file = next((f for f in config_files if f.exists()), None)
    if not config_file or not model_file.exists():
        print("❌ Model not found! Run: python download_mamba2_official.py")
        return
    
    try:
        start_time = time.time()
        config = MambaConfig(**json.load(open(config_file)))
        model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
        state_dict = torch.load(model_file, map_location="cpu")
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        model.eval()
        load_time = time.time() - start_time
        
        total_params = sum(p.numel() for p in model.parameters())
        mixer = model.backbone.layers[0].mixer
        print(f"✅ Loaded in {load_time:.2f}s: {total_params:,} params, d_state={mixer.d_state}, nheads={mixer.nheads}")
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"🎯 Generating: '{args.prompt}'")
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(args.max_tokens):
                outputs = model(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        generation_time = time.time() - start_time
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"📝 '{result}' ({args.max_tokens / generation_time:.1f} tok/s)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 