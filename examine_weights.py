import torch
import os
import sys
from collections import OrderedDict

# Load the checkpoint without safety restrictions
checkpoint = torch.load("./checkpoints/final_checkpoint/checkpoint.pt", weights_only=False)

# Print the structure to understand what we're working with
print("Checkpoint keys:", checkpoint.keys())

# Let's explore the structure deeper
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"\nKey: {key} (type: dict)")
        for subkey in value.keys():
            print(f"  - Subkey: {subkey}")
    elif isinstance(value, torch.Tensor):
        print(f"\nKey: {key} (type: tensor, shape: {value.shape})")
    elif isinstance(value, (list, tuple)):
        print(f"\nKey: {key} (type: {type(value)}, length: {len(value)})")
        if len(value) > 0:
            print(f"  - First element type: {type(value[0])}")
    else:
        print(f"\nKey: {key} (type: {type(value)})")
        try:
            # Try to print some attributes if it's an object
            if hasattr(value, "__dict__"):
                print(f"  - Attributes: {vars(value).keys()}")
        except:
            pass

# Look for any nested state dictionaries or model parameters
def find_state_dicts(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            # Check if this dict looks like a state dict (contains tensors)
            tensor_count = sum(1 for v in value.values() if isinstance(v, torch.Tensor))
            if tensor_count > 10:  # Arbitrary threshold to identify state dicts
                print(f"Potential state dict found at: {new_path} (contains {tensor_count} tensors)")
                
                # Print some tensor shapes for verification
                sample_count = 0
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and sample_count < 5:
                        print(f"  - {k}: {v.shape}")
                        sample_count += 1
            
            # Continue recursion
            find_state_dicts(value, new_path)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            find_state_dicts(item, f"{path}[{i}]")

print("\n\nSearching for potential state dictionaries...")
find_state_dicts(checkpoint)
