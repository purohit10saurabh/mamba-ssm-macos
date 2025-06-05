#!/usr/bin/env python3
import argparse
import json
import logging
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_downloaded_model(model_path, device="mps"):
    model_dir = Path(model_path)
    
    logger.info(f"🔍 Starting model loading from: {model_dir}")
    
    config_file = model_dir / "mamba-130m-config.json"
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return None, None
    
    with open(config_file) as f:
        config_data = json.load(f)
    
    logger.info(f"📋 Original config: {config_data}")
    
    if 'ssm_cfg' not in config_data: config_data['ssm_cfg'] = {}
    config_data['ssm_cfg']['layer'] = 'Mamba1'
    config_data['fused_add_norm'] = False
    
    logger.info(f"📋 Modified config: {config_data}")
    
    config = MambaConfig(**config_data)
    logger.info(f"✅ MambaConfig created successfully")
    
    print(f"🔧 Creating Mamba model...")
    logger.info("🔍 Checking triton availability...")
    try:
        import triton
        logger.info("✅ Triton module is available")
        TRITON_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Triton module not available: {e}")
        TRITON_AVAILABLE = False
    
    logger.info("🔍 Checking mamba module imports...")
    try:
        logger.info("🔍 Trying to import MambaLMHeadModel...")
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        logger.info("✅ MambaLMHeadModel imported successfully")
        
        logger.info("🔍 Creating MambaLMHeadModel instance...")
        model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
        logger.info("✅ MambaLMHeadModel created successfully")
        
    except ImportError as e:
        logger.error(f"❌ MambaLMHeadModel import error: {e}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        logger.info("🔄 Trying fallback model creation...")
        try:
            logger.info("🔍 Importing MixerModel...")
            from mamba_ssm.models.mixer_seq_simple import MixerModel
            logger.info("✅ MixerModel imported successfully")
            
            from torch import nn
            logger.info("✅ torch.nn imported successfully")
            
            class SimpleMambaLM(nn.Module):
                def __init__(self, config, device=None, dtype=None):
                    logger.info(f"🔍 Creating SimpleMambaLM with config: {vars(config)}")
                    super().__init__()
                    self.config = config
                    vocab_size = getattr(config, 'vocab_size', 50280)
                    logger.info(f"🔍 Using vocab_size: {vocab_size}")
                    
                    logger.info("🔍 Creating MixerModel backbone...")
                    self.backbone = MixerModel(
                        d_model=config.d_model,
                        n_layer=config.n_layer,
                        d_intermediate=getattr(config, 'd_intermediate', 0),
                        vocab_size=vocab_size,
                        ssm_cfg=getattr(config, 'ssm_cfg', {}),
                        device=device,
                        dtype=dtype,
                        fused_add_norm=False
                    )
                    logger.info("✅ MixerModel backbone created successfully")
                    
                    logger.info("🔍 Creating LM head...")
                    self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False, device=device, dtype=dtype)
                    logger.info("✅ LM head created successfully")
                
                def forward(self, input_ids, **kwargs):
                    logger.debug(f"🔍 Forward pass with input shape: {input_ids.shape}")
                    hidden_states = self.backbone(input_ids, **kwargs)
                    logits = self.lm_head(hidden_states)
                    return type('Output', (), {'logits': logits})()
                
                def generate(self, input_ids, max_length=50, temperature=0.8, do_sample=True, **kwargs):
                    logger.info(f"🔍 Generation with max_length={max_length}, temperature={temperature}")
                    self.eval()
                    with torch.no_grad():
                        generated = input_ids.clone()
                        for step in range(max_length - input_ids.shape[1]):
                            if step % 10 == 0:
                                logger.debug(f"🔍 Generation step {step}/{max_length - input_ids.shape[1]}")
                            outputs = self.forward(generated)
                            logits = outputs.logits[:, -1, :]
                            if temperature > 0 and do_sample:
                                logits = logits / temperature
                                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                            else:
                                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                            generated = torch.cat([generated, next_token], dim=1)
                    logger.info("✅ Generation completed")
                    return generated
                
                def load_state_dict(self, state_dict, strict=True):
                    logger.info(f"🔍 Loading state dict with {len(state_dict)} keys")
                    model_keys, state_keys = set(self.state_dict().keys()), set(state_dict.keys())
                    logger.info(f"🔍 Model has {len(model_keys)} keys, state_dict has {len(state_keys)} keys")
                    
                    new_state_dict, matched_keys = {}, 0
                    for key, value in state_dict.items():
                        if key in model_keys:
                            new_state_dict[key] = value
                            matched_keys += 1
                        elif key.replace('backbone.', '') in model_keys:
                            new_key = key.replace('backbone.', '')
                            new_state_dict[new_key] = value
                            matched_keys += 1
                            logger.debug(f"🔄 Mapped key: {key} -> {new_key}")
                        elif 'backbone.' + key in model_keys:
                            new_key = 'backbone.' + key
                            new_state_dict[new_key] = value
                            matched_keys += 1
                            logger.debug(f"🔄 Mapped key: {key} -> {new_key}")
                    
                    logger.info(f"✅ Matched {matched_keys}/{len(state_dict)} keys")
                    return super().load_state_dict(new_state_dict, strict=False)
            
            logger.info("🔍 Creating SimpleMambaLM instance...")
            model = SimpleMambaLM(config, device=device, dtype=torch.float32)
            logger.info("✅ SimpleMambaLM created successfully")
            
        except Exception as e2:
            logger.error(f"❌ Failed to create fallback model: {e2}")
            logger.error(f"❌ Full fallback traceback: {traceback.format_exc()}")
            return None, None
    
    model_file = model_dir / "mamba-130m-model.bin"
    if not model_file.exists():
        logger.error(f"❌ Model weights not found: {model_file}")
        return None, None
    
    logger.info(f"📦 Loading pre-trained weights from {model_file}...")
    try:
        logger.info("🔍 Loading state dict from file...")
        state_dict = torch.load(model_file, map_location="cpu")
        logger.info(f"✅ State dict loaded with {len(state_dict)} keys")
        
        logger.info("🔍 Loading state dict into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ State dict loaded: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
        if missing_keys: logger.warning(f"⚠️ Missing keys: {missing_keys[:5]}...")
        if unexpected_keys: logger.warning(f"⚠️ Unexpected keys: {unexpected_keys[:5]}...")
            
        model.eval()
        logger.info("✅ Model set to eval mode")
        
    except Exception as e:
        logger.error(f"❌ Error loading weights: {e}")
        logger.error(f"❌ Full weight loading traceback: {traceback.format_exc()}")
        return None, None
    
    # Load tokenizer
    logger.info("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        return None, None
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🎯 Model ready: {total_params:,} parameters")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, device="mps"):
    """Generate text using the pre-trained model."""
    logger.info(f"🚀 Starting text generation...")
    logger.info(f"🔍 Prompt: '{prompt}'")
    logger.info(f"🔍 Parameters: max_length={max_length}, temperature={temperature}, device={device}")
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.info(f"🔍 Input IDs shape: {input_ids.shape}")
    except Exception as e:
        logger.error(f"❌ Error encoding prompt: {e}")
        return None, 0
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        # Use the model's generation method
        try:
            logger.info("🔍 Trying built-in generation method...")
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            logger.info("✅ Built-in generation succeeded")
        except Exception as e:
            logger.warning(f"⚠️ Built-in generation failed: {e}")
            logger.info("🔄 Using simple greedy generation...")
            
            # Fallback to simple generation
            generated_sequence = input_ids.clone()
            for step in range(max_length - input_ids.shape[1]):
                if step % 10 == 0:
                    logger.debug(f"🔍 Generation step {step}")
                try:
                    outputs = model(generated_sequence)
                    logits = outputs.logits[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                except Exception as step_e:
                    logger.error(f"❌ Error at generation step {step}: {step_e}")
                    break
            output = generated_sequence
            logger.info("✅ Fallback generation completed")
    
    generation_time = time.time() - start_time
    logger.info(f"⏱️ Generation took {generation_time:.2f} seconds")
    
    try:
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"✅ Text decoded successfully, length: {len(generated_text)}")
    except Exception as e:
        logger.error(f"❌ Error decoding text: {e}")
        return None, generation_time
    
    return generated_text, generation_time

def main():
    parser = argparse.ArgumentParser(description="Run downloaded Mamba model")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Text prompt")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Device")
    parser.add_argument("--model-dir", type=str, default="./models", help="Downloaded models directory")
    
    args = parser.parse_args()
    
    print(f"🍎 Loading downloaded Mamba model on {args.device}...")
    
    model, tokenizer = load_downloaded_model(args.model_dir, args.device)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print(f"💭 Input prompt: '{args.prompt}'")
    
    generated_text, generation_time = generate_text(
        model, tokenizer, args.prompt, args.max_length, args.temperature, args.device
    )
    
    print(f"\n📝 Generated text ({generation_time:.2f}s):")
    print(f"'{generated_text}'")
    
    # Calculate performance metrics
    tokens_generated = len(generated_text.split()) - len(args.prompt.split())
    if tokens_generated > 0 and generation_time > 0:
        tokens_per_sec = tokens_generated / generation_time
        print(f"\n📊 Performance: {tokens_per_sec:.1f} tokens/sec")

if __name__ == "__main__":
    main() 