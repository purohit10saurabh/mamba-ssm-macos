#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mamba_macos import generate_text_with_model, get_device, load_and_prepare_model


def run_generation(model, tokenizer, args, model_name):
    generated_text = generate_text_with_model(
        model, tokenizer, args.prompt, args.device, args.max_length, args.temperature
    )
    print(f"\n📝 Generated text:\n'{generated_text}'")


def run_model(model_name, args):
    print(f"🍎 Loading downloaded {model_name.title()} model on {args.device}...")

    try:
        success, model, tokenizer = load_and_prepare_model(
            model_name, args.model_dir, args.device
        )
        if not success:
            return False

        print(f"💭 Input prompt: '{args.prompt}'")
        run_generation(model, tokenizer, args, model_name)
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Mamba models")
    parser.add_argument("model", choices=["mamba1", "mamba2"], help="Model to run")
    parser.add_argument("--prompt", default="The future of AI is", help="Text prompt")
    parser.add_argument(
        "--max-length", type=int, default=50, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--device", default=get_device(), help="Device")
    parser.add_argument(
        "--model-dir", default="./models", help="Downloaded models directory"
    )

    args = parser.parse_args()
    if args.model == "mamba2" and args.max_length == 50:
        args.max_length = 8

    success = run_model(args.model, args)
    print("✅ SUCCESS" if success else "❌ FAILED")


if __name__ == "__main__":
    main()
