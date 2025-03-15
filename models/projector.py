# models/projector.py
import torch
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Test function for projector
def test_projector():
    # Whisper encoder output dim (e.g., 1024)
    input_dim = 1024
    # LLaMA embedding dim (e.g., 4096 for LLaMA)
    output_dim = 4096
    
    projector = AudioProjector(input_dim, output_dim)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = projector(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, output_dim)
    # Check parameters are trainable
    for param in projector.parameters():
        assert param.requires_grad
    
    print("Projector implementation successful")
