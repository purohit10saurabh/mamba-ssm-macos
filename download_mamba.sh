#!/bin/bash

DEST_DIR="./models"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "🍎 Downloading Mamba 130M..."

# Mamba 130M model
curl -L https://huggingface.co/state-spaces/mamba-130m/raw/main/config.json -o ./mamba-130m-config.json
curl -L https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin -o ./mamba-130m-model.bin

echo "✅ Download complete! Run: python -m examples.07_downloaded_model_demo" 