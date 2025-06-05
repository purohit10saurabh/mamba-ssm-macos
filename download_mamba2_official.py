#!/usr/bin/env python3
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def download_mamba2():
    local_dir = "models/official_mamba2"
    print(f"Downloading state-spaces/mamba2-130m to {local_dir}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download("state-spaces/mamba2-130m", local_dir=local_dir, 
                         resume_download=True, local_dir_use_symlinks=False)
        print("✅ Download complete")
        
        # Create complete Mamba2 config with all required parameters
        config_path = f"{local_dir}/config.json"
        if Path(config_path).exists():
            official_config = json.load(open(config_path))
            # Add missing Mamba2 parameters required for weight loading
            official_config["ssm_cfg"] = {
                "layer": "Mamba2", "d_state": 128, "d_conv": 4, 
                "expand": 2, "headdim": 64, "ngroups": 1
            }
            # Apple Silicon compatibility - disable fused operations that require Triton
            official_config["fused_add_norm"] = False
            with open("models/mamba2_config.json", "w") as f:
                json.dump(official_config, f, indent=2)
            print("✅ Config file created: models/mamba2_config.json (complete Mamba2 parameters)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    download_mamba2() 