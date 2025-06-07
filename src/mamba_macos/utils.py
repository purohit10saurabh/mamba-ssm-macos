#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
    config_data["fused_add_norm"] = False
    return MambaConfig(**config_data)


def create_mamba2_config(config_data):
    config_data["ssm_cfg"] = {
        "layer": "Mamba2",
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
    }
    config_data["fused_add_norm"] = False
    return MambaConfig(**config_data)


def find_config(config_paths, config_creator):
    for config_path in config_paths:
        config_data = load_config_file(config_path)
        if config_data:
            return config_creator(config_data), config_path
    return None, None


def create_model_with_fallback(config, device):
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel(config, device=device, dtype=torch.float32)
    except ImportError:
        return create_fallback_model(config, device)


def create_fallback_model(config, device):
    from torch import nn

    from mamba_ssm.models.mixer_seq_simple import MixerModel

    class SimpleMambaLM(nn.Module):
        def __init__(self, config, device=None, dtype=None):
            super().__init__()
            vocab_size = getattr(config, "vocab_size", 50280)
            self.backbone = MixerModel(
                d_model=config.d_model,
                n_layer=config.n_layer,
                d_intermediate=getattr(config, "d_intermediate", 0),
                vocab_size=vocab_size,
                ssm_cfg=getattr(config, "ssm_cfg", {}),
                device=device,
                dtype=dtype,
                fused_add_norm=False,
            )
            self.lm_head = nn.Linear(
                config.d_model, vocab_size, bias=False, device=device, dtype=dtype
            )

        def forward(self, input_ids, **kwargs):
            return type(
                "Output",
                (),
                {"logits": self.lm_head(self.backbone(input_ids, **kwargs))},
            )()

    return SimpleMambaLM(config, device=device, dtype=torch.float32)


def create_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_weights(model, weight_paths):
    for weight_path in weight_paths:
        if Path(weight_path).exists():
            state_dict = torch.load(weight_path, map_location="cpu")
            missing_keys, _ = model.load_state_dict(state_dict, strict=False)
            model.eval()
            return len(missing_keys)
    return -1


def generate_text_with_model(model, tokenizer, prompt, device, max_length, temperature):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = input_ids.clone()
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
