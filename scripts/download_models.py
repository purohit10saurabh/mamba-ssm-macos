#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


def create_config(source_config_path, output_config_path, model_type):
    config_data = json.load(open(source_config_path))

    if model_type == "mamba1":
        if "ssm_cfg" not in config_data:
            config_data["ssm_cfg"] = {}
        if "layer" not in config_data["ssm_cfg"]:
            config_data["ssm_cfg"]["layer"] = "Mamba1"
    else:
        config_data["ssm_cfg"] = {
            "layer": "Mamba2",
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "headdim": 64,
            "ngroups": 1,
        }

    config_data["fused_add_norm"] = False
    with open(output_config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    return output_config_path


def download_model(model_type, model_dir):
    local_dir = f"{model_dir}/{model_type}"
    print(f"Downloading state-spaces/{model_type}-130m to {local_dir}")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    if model_type == "mamba1":
        print("📦 Downloading config.json...")
        config_path = hf_hub_download(
            "state-spaces/mamba-130m",
            "config.json",
            local_dir=local_dir,
            resume_download=True,
        )
        print("📦 Downloading pytorch_model.bin...")
        model_path = hf_hub_download(
            "state-spaces/mamba-130m",
            "pytorch_model.bin",
            local_dir=local_dir,
            resume_download=True,
        )

        print("🔄 Renaming files...")
        final_config = f"{local_dir}/{model_type}-130m-config.json"
        final_model = f"{local_dir}/{model_type}-130m-model.bin"
        Path(config_path).rename(final_config)
        Path(model_path).rename(final_model)
        downloaded_files = [
            f"{model_type}-130m-config.json",
            f"{model_type}-130m-model.bin",
        ]
    else:
        print("📦 Downloading all files...")
        snapshot_download(
            "state-spaces/mamba2-130m",
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False,
        )

        print("🔄 Renaming files...")
        original_config = f"{local_dir}/config.json"
        final_config = f"{local_dir}/{model_type}-130m-config.json"
        final_model = f"{local_dir}/{model_type}-130m-model.bin"

        if Path(original_config).exists():
            create_config(original_config, final_config, model_type)
        Path(f"{local_dir}/pytorch_model.bin").rename(final_model)
        downloaded_files = [
            f"{model_type}-130m-config.json",
            f"{model_type}-130m-model.bin",
        ]

    print("✅ Download complete")
    print(f"📁 Files saved to: {local_dir}/")
    for file_name in downloaded_files:
        print(f"   - {file_name}")


def main():
    parser = argparse.ArgumentParser(description="Download Mamba models")
    parser.add_argument(
        "model", choices=["mamba1", "mamba2", "both"], help="Model to download"
    )
    parser.add_argument(
        "--model-dir", default="./models", help="Downloaded models directory"
    )
    args = parser.parse_args()

    try:
        if args.model in ["mamba1", "both"]:
            download_model("mamba1", args.model_dir)
        if args.model in ["mamba2", "both"]:
            download_model("mamba2", args.model_dir)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("✅ SUCCESS" if main() else "❌ FAILED")
