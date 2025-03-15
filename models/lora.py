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
        
        # init A with zeros
        nn.init.zeros_(self.lora_A)
        # init B with small random values
        nn.init.normal_(self.lora_B, std=0.01)
    
    def forward(self, x):
        return (x @ (self.lora_B @ self.lora_A).T) * self.scaling

def apply_lora_to_llama(llama_model, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA to specific modules in LLaMA
    """
    if target_modules is None:
        # apply to q_proj, k_proj, v_proj, and mlp layers
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    lora_layers = {}
    
    for name, module in llama_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target_name in name for target_name in target_modules):
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
                lora_layers[name] = lora
    
    return lora_layers

def lora_forward_hook(module, input, output, lora_layer):
    """Add LoRA output to the original linear layer output"""
    return output + lora_layer(input[0])

def test_lora():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    
    lora = LoRALayer(10, 20, rank=4)
    
    for param in lora.parameters():
        assert param.requires_grad
    
    x = torch.randn(2, 10)
    original_output = model(x)
    
    lora_output = lora(x)
    combined_output = original_output + lora_output
    
    assert lora_output.shape == original_output.shape
    assert combined_output.shape == original_output.shape
    
    print("LoRA implementation successful")
