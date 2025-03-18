#!/usr/bin/env python
"""
Upload a model checkpoint to the Hugging Face Hub.

This script uploads a model checkpoint to Hugging Face Hub,
along with tokenizer, model card, and other necessary files.

Usage:
    python upload_to_huggingface.py --checkpoint_path path/to/final_checkpoint \
        --model_name your-username/model-name \
        --description "Brief description of your model"
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import tempfile
import logging

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, upload_folder, login
except ImportError:
    print("Please install the required packages with:")
    print("pip install transformers huggingface-hub torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model_card(repo_id, description, dataset_info=None):
    """Create a model card markdown file with basic information."""
    dataset_text = f"- Trained on: {dataset_info}\n" if dataset_info else ""
    
    model_card = f"""---
language: en
license: apache-2.0
tags:
- audio
- speech
- transcription
- librispeech
datasets:
- librispeech_asr
---

# {repo_id.split('/')[-1]}

## Model Description

{description}

## Training Details

{dataset_text}
- This model was fine-tuned on audio transcription tasks.
- The base architecture is derived from Llama.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Example usage
prompt = "Transcribe this audio:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Limitations

This model is designed specifically for audio transcription tasks and may not perform well on other tasks.
"""
    
    return model_card

def prepare_model_for_upload(checkpoint_path, output_dir, tokenizer_path=None):
    """
    Load the checkpoint and prepare it for upload to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint file or directory
        output_dir: Directory to save the prepared model files
        tokenizer_path: Optional path to the tokenizer, if different from checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Determine if we're dealing with a directory or a file
    is_directory = os.path.isdir(checkpoint_path)
    
    if is_directory:
        # If it's a directory, check for PyTorch or HF format
        if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) or \
           os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            # This is already in HF format, just copy it
            logger.info("Checkpoint is already in Hugging Face format")
            for item in os.listdir(checkpoint_path):
                src = os.path.join(checkpoint_path, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            return
    
    # Check if this is a PyTorch checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info("Loaded PyTorch checkpoint")
        
        # Try to determine the model architecture
        model_type = "unknown"
        if "model_type" in checkpoint:
            model_type = checkpoint["model_type"]
        elif "config" in checkpoint and "model_type" in checkpoint["config"]:
            model_type = checkpoint["config"]["model_type"]
        
        logger.info(f"Detected model type: {model_type}")
        
        # Load and save the model using transformers
        if model_type != "unknown":
            # For known model types, try to use the appropriate class
            try:
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                model.save_pretrained(output_dir)
                logger.info(f"Saved model to {output_dir}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Copying raw checkpoint instead")
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy2(checkpoint_path, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            # If we can't determine the type, just copy the raw checkpoint
            logger.info("Copying raw checkpoint")
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(checkpoint_path, os.path.join(output_dir, "pytorch_model.bin"))
            
            # Create a basic config file
            config = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"]
            }
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.error("Please make sure the checkpoint is in a valid format.")
        sys.exit(1)
    
    # Handle tokenizer
    try:
        if tokenizer_path is None:
            tokenizer_path = checkpoint_path if is_directory else os.path.dirname(checkpoint_path)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved tokenizer to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not load/save tokenizer: {e}")
        logger.warning("You may need to upload the tokenizer separately.")

def upload_to_huggingface(model_path, repo_id, token=None, description=None, dataset_info=None):
    """
    Upload the model to Hugging Face Hub.
    
    Args:
        model_path: Path to the prepared model directory
        repo_id: Hugging Face repository ID in the format "username/model-name"
        token: Hugging Face API token
        description: Model description
        dataset_info: Information about the dataset used for training
    """
    # Check for token in environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    # If token is still None, try to use stored credentials
    if token:
        login(token=token, add_to_git_credential=True)
    
    # Create a temporary directory for preparation
    with tempfile.TemporaryDirectory() as tmp_dir:
        # If model_path is a file, we need to convert it
        if os.path.isfile(model_path):
            logger.info(f"Converting single checkpoint file: {model_path}")
            prepare_model_for_upload(model_path, tmp_dir)
            upload_path = tmp_dir
        elif os.path.isdir(model_path):
            # Check if the directory is already in HF format
            if os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(model_path, "model.safetensors")) or \
               os.path.exists(os.path.join(model_path, "config.json")):
                logger.info("Directory appears to be in HF format already")
                upload_path = model_path
            else:
                # Need to convert
                logger.info("Converting checkpoint directory")
                prepare_model_for_upload(model_path, tmp_dir)
                upload_path = tmp_dir
        else:
            logger.error(f"Model path {model_path} does not exist or is not accessible")
            sys.exit(1)
        
        # Create and save model card
        if description:
            model_card = create_model_card(repo_id, description, dataset_info)
            with open(os.path.join(upload_path, "README.md"), "w") as f:
                f.write(model_card)
            logger.info("Created model card")
        
        # Upload to HF
        logger.info(f"Uploading model to {repo_id}")
        try:
            api = HfApi()
            response = upload_folder(
                folder_path=upload_path,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["*.pyc", ".git*", ".DS_Store"],
                create_repo=True
            )
            logger.info(f"Upload successful! Model is available at: https://huggingface.co/{repo_id}")
            return response
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Upload a model checkpoint to Hugging Face Hub")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the model checkpoint file or directory")
    parser.add_argument("--model_name", required=True, help="Name for the model on HF Hub (username/model-name)")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env variable)")
    parser.add_argument("--description", default="An audio transcription model trained on LibriSpeech", 
                        help="Short description of the model")
    parser.add_argument("--dataset_info", default="LibriSpeech ASR corpus", 
                        help="Information about the dataset used for training")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer files (if different from checkpoint)")
    parser.add_argument("--convert_only", action="store_true", 
                        help="Only convert the checkpoint to HF format without uploading")
    parser.add_argument("--output_dir", help="Output directory for converted model (used with --convert_only)")
    
    args = parser.parse_args()
    
    # Validate args
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint path does not exist: {args.checkpoint_path}")
        sys.exit(1)
    
    if args.convert_only:
        if not args.output_dir:
            logger.error("--output_dir is required when using --convert_only")
            sys.exit(1)
        
        os.makedirs(args.output_dir, exist_ok=True)
        prepare_model_for_upload(args.checkpoint_path, args.output_dir, args.tokenizer_path)
        logger.info(f"Model converted and saved to {args.output_dir}")
    else:
        upload_to_huggingface(
            args.checkpoint_path, 
            args.model_name, 
            args.token, 
            args.description, 
            args.dataset_info
        )

if __name__ == "__main__":
    main()
