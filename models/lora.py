# models/lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize A with zeros
        nn.init.zeros_(self.lora_A)
        # Initialize B with small random values
        nn.init.normal_(self.lora_B, std=0.01)
    
    def forward(self, x):
        return (x @ (self.lora_B @ self.lora_A).T) * self.scaling

def apply_lora_to_llama(llama_model, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA to specific modules in LLaMA
    """
    if target_modules is None:
        # Default: apply to q_proj, k_proj, v_proj, and mlp layers
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    lora_layers = {}
    
    # Find all linear layers and apply LoRA to targeted ones
    for name, module in llama_model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this is a target module
            if any(target_name in name for target_name in target_modules):
                # Create a LoRA layer
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
                lora_layers[name] = lora
    
    return lora_layers

# Original forward hook for linear layers in LLaMA
def lora_forward_hook(module, input, output, lora_layer):
    """Add LoRA output to the original linear layer output"""
    return output + lora_layer(input[0])

# Test function for LoRA
def test_lora():
    # Create a small model with a linear layer
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    
    # Create a LoRA layer
    lora = LoRALayer(10, 20, rank=4)
    
    # Check parameters are trainable
    for param in lora.parameters():
        assert param.requires_grad
    
    # Test forward
    x = torch.randn(2, 10)
    original_output = model(x)
    
    # Add LoRA output
    lora_output = lora(x)
    combined_output = original_output + lora_output
    
    # Check shapes
    assert lora_output.shape == original_output.shape
    assert combined_output.shape == original_output.shape
    
    print("LoRA implementation successful")
