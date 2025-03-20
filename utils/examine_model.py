import torch
import os
import json
from collections import OrderedDict

# Load the checkpoint without safety restrictions
checkpoint = torch.load("./checkpoints/final_checkpoint/checkpoint.pt", weights_only=False)

# Let's examine the model key specifically
if 'model' in checkpoint:
    model_dict = checkpoint['model']
    print("Model keys:", model_dict.keys())
    
    # Check the first level keys
    for key, value in model_dict.items():
        if isinstance(value, dict):
            print(f"\nKey: {key} (type: dict, keys: {value.keys()})")
        elif isinstance(value, torch.nn.Module):
            print(f"\nKey: {key} (type: nn.Module, class: {value.__class__.__name__})")
            # Try to get state_dict from the module
            try:
                state_dict = value.state_dict()
                print(f"  - State dict size: {len(state_dict)} items")
                # Print a few sample parameter shapes
                for i, (param_name, param) in enumerate(state_dict.items()):
                    if i >= 5:  # Only show first 5
                        break
                    print(f"  - {param_name}: {param.shape}")
            except Exception as e:
                print(f"  - Error getting state_dict: {e}")
        elif isinstance(value, torch.Tensor):
            print(f"\nKey: {key} (type: tensor, shape: {value.shape})")
        else:
            print(f"\nKey: {key} (type: {type(value)})")

    # Check if there are LoRA layers (common in fine-tuned models)
    if 'lora_layers' in model_dict:
        print("\nExamining LoRA layers:")
        lora_layers = model_dict['lora_layers']
        if isinstance(lora_layers, dict):
            print(f"LoRA layers keys: {lora_layers.keys()}")
            # Try to dig into the LoRA structure
            for lora_key, lora_value in lora_layers.items():
                print(f"  - {lora_key}: {type(lora_value)}")
                if hasattr(lora_value, 'state_dict'):
                    try:
                        lora_state = lora_value.state_dict()
                        print(f"    - State dict size: {len(lora_state)} items")
                        # Print a few keys
                        for i, key in enumerate(lora_state.keys()):
                            if i >= 5:  # Only show first 5
                                break
                            print(f"    - {key}")
                    except Exception as e:
                        print(f"    - Error getting state_dict: {e}")

# Extract some useful metadata from args
if 'args' in checkpoint:
    args = checkpoint['args']
    print("\nImportant args:")
    print(f"  - llama_path: {getattr(args, 'llama_path', 'N/A')}")
    print(f"  - whisper_path: {getattr(args, 'whisper_path', 'N/A')}")
    print(f"  - lora_rank: {getattr(args, 'lora_rank', 'N/A')}")
    print(f"  - model architecture details: {vars(args)}")

# Save args to a JSON file for reference
if 'args' in checkpoint:
    with open("model_args.json", "w") as f:
        # Convert Namespace to dict
        args_dict = vars(checkpoint['args'])
        json.dump(args_dict, f, indent=2, default=str)
    print("\nSaved args to model_args.json")

# Print training progress info
print("\nTraining progress:")
print(f"  - Current epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  - Current step: {checkpoint.get('step', 'N/A')}")
