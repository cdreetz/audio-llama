import torch
import os

# Load the checkpoint without safety restrictions
checkpoint = torch.load("./checkpoints/final_checkpoint/checkpoint.pt", weights_only=False)

# Print the structure to understand what we're working with
print("Checkpoint keys:", checkpoint.keys())

# Typically, the model weights are stored in a key like 'model_state_dict' or 'state_dict'
# Extract just the model weights
if 'model_state_dict' in checkpoint:
    model_weights = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    model_weights = checkpoint['state_dict']
else:
    # List all keys to help identify where the weights are
    print("Available keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"- {key}")
    raise ValueError("Could not identify model weights in checkpoint")

# Save just the weights in a new file
torch.save(model_weights, "./checkpoints/final_checkpoint/model_weights_only.pt")
print("Saved model weights to model_weights_only.pt")
