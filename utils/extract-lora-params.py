#!/usr/bin/env python
"""
Inspect the LoRA parameters in the checkpoint.

This script examines the internal structure of the LoRA layers to help
understand how to properly extract them.

Usage:
    python inspect_lora_params.py
"""

import os
import torch
import json
import logging
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to checkpoint
CHECKPOINT_PATH = "./checkpoints/final_checkpoint/checkpoint.pt"

def inspect_lora_layers():
    # Load checkpoint
    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    
    if 'model' not in checkpoint or 'lora_layers' not in checkpoint['model']:
        logger.error("No LoRA layers found in checkpoint")
        return
    
    # Get the LoRA layers
    lora_layers = checkpoint['model']['lora_layers']
    
    # Sample a few layers to inspect their structure
    sample_layer_keys = list(lora_layers.keys())[:3]  # Just look at the first 3
    
    logger.info(f"Inspecting {len(sample_layer_keys)} sample LoRA layers")
    
    for layer_key in sample_layer_keys:
        layer = lora_layers[layer_key]
        logger.info(f"Layer: {layer_key}")
        
        if isinstance(layer, dict) or isinstance(layer, OrderedDict):
            # Inspect keys and values
            logger.info(f"  Type: {type(layer).__name__}")
            logger.info(f"  Keys: {list(layer.keys())}")
            
            # Inspect each item in this layer
            for key, value in layer.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  - {key}: Tensor with shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, (int, float)):
                    logger.info(f"  - {key}: {type(value).__name__} with value {value}")
                else:
                    logger.info(f"  - {key}: {type(value).__name__}")
        
        elif isinstance(layer, torch.nn.Module):
            logger.info(f"  Type: torch.nn.Module (class: {layer.__class__.__name__})")
            # Try to get state dict
            try:
                state_dict = layer.state_dict()
                logger.info(f"  State dict keys: {list(state_dict.keys())}")
                
                # Show shapes of parameters
                for param_name, param in state_dict.items():
                    logger.info(f"  - {param_name}: Tensor with shape {param.shape}, dtype {param.dtype}")
            except Exception as e:
                logger.error(f"  Error getting state_dict: {e}")
        
        elif isinstance(layer, torch.Tensor):
            logger.info(f"  Type: torch.Tensor with shape {layer.shape}, dtype {layer.dtype}")
        
        else:
            logger.info(f"  Type: {type(layer).__name__}")
    
    # Check for LoRA hyperparameters in args
    if 'args' in checkpoint:
        args = checkpoint['args']
        lora_rank = getattr(args, 'lora_rank', None)
        logger.info(f"LoRA rank from args: {lora_rank}")
    
    # Count total parameters
    total_params = 0
    unique_layer_types = set()
    
    # Count parameters by type
    types_count = {}
    
    for layer_key, layer in lora_layers.items():
        layer_type = type(layer).__name__
        unique_layer_types.add(layer_type)
        
        types_count[layer_type] = types_count.get(layer_type, 0) + 1
        
        if isinstance(layer, dict) or isinstance(layer, OrderedDict):
            for key, value in layer.items():
                if isinstance(value, torch.Tensor):
                    total_params += value.numel()
    
    logger.info(f"Total LoRA parameters: {total_params:,}")
    logger.info(f"Unique layer types: {unique_layer_types}")
    logger.info(f"Layer type counts: {types_count}")
    
    # Write a sample layer to file for detailed inspection
    if sample_layer_keys:
        sample_layer = lora_layers[sample_layer_keys[0]]
        
        try:
            if isinstance(sample_layer, dict) or isinstance(sample_layer, OrderedDict):
                # Convert tensors to lists for JSON serialization
                serializable_layer = {}
                for k, v in sample_layer.items():
                    if isinstance(v, torch.Tensor):
                        serializable_layer[k] = {
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "device": str(v.device),
                            "sample_values": v.flatten()[:5].tolist() if v.numel() > 0 else []
                        }
                    else:
                        serializable_layer[k] = str(v)
                
                with open("sample_lora_layer.json", "w") as f:
                    json.dump(serializable_layer, f, indent=2)
                
                logger.info(f"Saved sample layer to sample_lora_layer.json")
        except Exception as e:
            logger.error(f"Error saving sample layer: {e}")

if __name__ == "__main__":
    inspect_lora_layers()
