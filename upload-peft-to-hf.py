#!/usr/bin/env python
"""
Upload PEFT adapter to Hugging Face Hub.

This script uploads the converted PEFT adapter to Hugging Face Hub.

Usage:
    python upload_peft_to_huggingface.py --model_name your_username/model_name --token YOUR_HF_TOKEN
"""

import os
import argparse
import logging
from huggingface_hub import HfApi, login, create_repo, upload_folder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Upload PEFT adapter to Hugging Face Hub")
    parser.add_argument("--model_dir", default="./peft_model", help="Path to the PEFT model directory")
    parser.add_argument("--model_name", required=True, help="Name for the model on HF Hub (username/model-name)")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env variable)")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token if args.token else os.environ.get("HF_TOKEN")
    
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.bin", "README.md"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.model_dir, f))]
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return
    
    # Login to Hugging Face if token is provided
    if token:
        logger.info("Logging in to Hugging Face Hub")
        login(token=token, add_to_git_credential=True)
    else:
        logger.warning("No token provided, assuming you're already logged in")
    
    # Create repository if it doesn't exist
    try:
        logger.info(f"Creating repository: {args.model_name}")
        api = HfApi()
        create_repo(args.model_name, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        return
    
    # Upload files
    try:
        logger.info(f"Uploading files to {args.model_name}")
        upload_folder(
            folder_path=args.model_dir,
            repo_id=args.model_name,
            repo_type="model",
            ignore_patterns=["*.pyc", ".git*", ".DS_Store"],
            token=token
        )
        logger.info(f"Upload successful! Model available at: https://huggingface.co/{args.model_name}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return

if __name__ == "__main__":
    main()
