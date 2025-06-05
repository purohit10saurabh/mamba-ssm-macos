#!/usr/bin/env python3
"""Model loading and preparation utilities for Mamba SSM on macOS."""

from .utils import (create_mamba1_config, create_mamba2_config,
                    create_model_with_fallback, create_tokenizer, find_config,
                    load_model_weights)


def get_model_paths(model_name, model_dir):
    """Get config and weight file paths for a given model."""
    config_paths = [f"{model_dir}/{model_name}/{model_name}-130m-config.json"]
    weight_paths = [f"{model_dir}/{model_name}/{model_name}-130m-model.bin"]
    config_creator = create_mamba1_config if model_name == "mamba1" else create_mamba2_config
    return config_paths, weight_paths, config_creator


def load_and_prepare_model(model_name, model_dir, device):
    """Load and prepare a Mamba model for inference.
    
    Args:
        model_name: Name of the model ('mamba1' or 'mamba2')
        model_dir: Directory containing model files
        device: Device to load model on ('cpu', 'mps', 'cuda')
        
    Returns:
        Tuple of (success, model, tokenizer)
    """
    config_paths, weight_paths, config_creator = get_model_paths(model_name, model_dir)
    
    config, config_file = find_config(config_paths, config_creator)
    if not config:
        print("❌ No config file found")
        return False, None, None
    
    if config_file: 
        print(f"Using config: {config_file}")
    
    model = create_model_with_fallback(config, device)
    tokenizer = create_tokenizer()
    
    missing_keys = load_model_weights(model, weight_paths)
    if missing_keys == -1:
        print("❌ No weight file found")
        return False, None, None
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model ready: {total_params:,} parameters")
    
    return True, model, tokenizer 