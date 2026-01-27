import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_config_file(config_path):
    if not Path(config_path).exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def create_mamba1_config(config_data):
    if "ssm_cfg" not in config_data:
        config_data["ssm_cfg"] = {}
    if "layer" not in config_data["ssm_cfg"]:
        config_data["ssm_cfg"]["layer"] = "Mamba1"
    return MambaConfig(**config_data)


def create_mamba2_config(config_data):
    config_data["ssm_cfg"] = {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64, "ngroups": 1}
    return MambaConfig(**config_data)


def load_and_prepare_model(model_name, model_dir, device):
    config_path = f"{model_dir}/{model_name}/{model_name}-130m-config.json"
    weight_path = f"{model_dir}/{model_name}/{model_name}-130m-model.bin"
    config_creator = create_mamba1_config if model_name == "mamba1" else create_mamba2_config

    config_data = load_config_file(config_path)
    if not config_data:
        print("❌ No config file found")
        return False, None, None

    print(f"Using config: {config_path}")
    config = config_creator(config_data)
    model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
    tokenizer = create_tokenizer()

    if not Path(weight_path).exists():
        print("❌ No weight file found")
        return False, None, None

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model ready: {total_params:,} parameters")
    return True, model, tokenizer


def generate_text_with_model(model, tokenizer, prompt, device, max_length, temperature, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = input_ids.clone()
        for _ in range(max_length - input_ids.shape[1]):
            logits = model(generated).logits[:, -1, :]
            if temperature > 0:
                next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0], skip_special_tokens=True)
