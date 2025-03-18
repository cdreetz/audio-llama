#!/usr/bin/env python
"""
Convert Audio-LLaMA LoRA checkpoint to Hugging Face PEFT format.

This script extracts LoRA adapter weights from the Audio-LLaMA checkpoint
and converts them to the Hugging Face PEFT format for easy sharing and usage.

Usage:
    python convert_lora_to_huggingface.py
"""

import os
import torch
import json
from collections import OrderedDict
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_PATH = "./checkpoints/final_checkpoint/checkpoint.pt"
OUTPUT_DIR = "./peft_model"
CONFIG_FILE = "adapter_config.json"

def extract_lora_state_dict(checkpoint):
    """Extract LoRA weights from the checkpoint."""
    state_dict = {}
    
    if 'model' not in checkpoint or 'lora_layers' not in checkpoint['model']:
        logger.error("This checkpoint does not contain LoRA layers")
        return None
    
    # Extract LoRA layers
    lora_dict = checkpoint['model']['lora_layers']
    
    # Process each LoRA layer
    for layer_name, layer_dict in lora_dict.items():
        # Expect each layer_dict to be an OrderedDict containing specific LoRA keys
        # We need to convert these to the PEFT format
        
        # Standard LoRA keys to look for
        lora_keys = ["lora_A", "lora_B", "scaling"]
        
        if isinstance(layer_dict, OrderedDict):
            # Extract keys from this layer
            for key in layer_dict.keys():
                # For each LoRA parameter in the layer
                if key in lora_keys:
                    peft_key = f"{layer_name}.{key}"
                    state_dict[peft_key] = layer_dict[key]
    
    return state_dict

def create_peft_config(checkpoint):
    """Create a PEFT adapter config from the checkpoint args."""
    args = checkpoint.get('args', None)
    
    if args is None:
        logger.warning("No args found in checkpoint, using default values")
    
    # Get LoRA rank
    lora_rank = getattr(args, 'lora_rank', 32) if args else 32
    
    # Extract base model name
    base_model_name = getattr(args, 'llama_path', 'meta-llama/Llama-3.2-3B-Instruct') if args else 'meta-llama/Llama-3.2-3B-Instruct'
    
    # Create adapter config
    config = {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "enable_lora": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": lora_rank,  # Common convention is to set alpha to rank
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "task_type": "CAUSAL_LM"
    }
    
    return config

def create_readme(checkpoint):
    """Create a README file with model information."""
    args = checkpoint.get('args', None)
    
    # Extract information
    base_model = getattr(args, 'llama_path', 'meta-llama/Llama-3.2-3B-Instruct') if args else 'meta-llama/Llama-3.2-3B-Instruct'
    whisper_model = getattr(args, 'whisper_path', 'openai/whisper-large-v3-turbo') if args else 'openai/whisper-large-v3-turbo'
    lora_rank = getattr(args, 'lora_rank', 32) if args else 32
    
    readme = f"""---
language: en
license: apache-2.0
tags:
- audio
- speech
- transcription
- librispeech
- llama-3
datasets:
- librispeech_asr
---

# Audio-LLaMA: LoRA Adapter for Audio Transcription

This model is a LoRA adapter fine-tuned on audio transcription tasks. It requires the Llama base model to be used.

## Model Details

- **Base Model**: {base_model}
- **Audio Model**: {whisper_model}
- **LoRA Rank**: {lora_rank}
- **Task**: Audio transcription from LibriSpeech dataset
- **Training Framework**: PEFT (Parameter-Efficient Fine-Tuning)

## Usage

This is a PEFT (LoRA) adapter that needs to be combined with the base Llama model to work:

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the LoRA configuration
config = PeftConfig.from_pretrained("cdreetz/audio-llama")

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, "cdreetz/audio-llama")

# Run inference
prompt = "Transcribe this audio:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

This model was fine-tuned using LoRA on audio transcription tasks. It starts with a Llama 3 base model and uses Whisper-processed audio features for audio understanding.

## Limitations

This model requires special code for audio processing with Whisper before passing to the Llama model. See the [Audio-LLaMA repository](https://github.com/cdreetz/audio-llama) for full usage instructions.
"""
    
    return readme

def main():
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load checkpoint with weights_only=False to bypass safety restrictions
        logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        logger.info("Checkpoint loaded successfully")
        
        # Extract LoRA state dict
        logger.info("Extracting LoRA weights")
        lora_state_dict = extract_lora_state_dict(checkpoint)
        
        if lora_state_dict is None:
            logger.error("Failed to extract LoRA weights")
            return
        
        logger.info(f"Extracted {len(lora_state_dict)} LoRA parameters")
        
        # Create adapter config
        logger.info("Creating PEFT adapter config")
        config = create_peft_config(checkpoint)
        
        # Save adapter config
        with open(os.path.join(OUTPUT_DIR, CONFIG_FILE), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save the state dict
        logger.info("Saving adapter state dict")
        torch.save(lora_state_dict, os.path.join(OUTPUT_DIR, "adapter_model.bin"))
        
        # Create and save README
        logger.info("Creating README")
        readme = create_readme(checkpoint)
        with open(os.path.join(OUTPUT_DIR, "README.md"), 'w') as f:
            f.write(readme)
        
        # Save original args for reference
        if 'args' in checkpoint:
            with open(os.path.join(OUTPUT_DIR, "original_args.json"), 'w') as f:
                args_dict = vars(checkpoint['args'])
                json.dump(args_dict, f, indent=2, default=str)
        
        logger.info(f"Conversion complete! PEFT adapter saved to {OUTPUT_DIR}")
        logger.info("You can now upload this directory to Hugging Face Hub")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
