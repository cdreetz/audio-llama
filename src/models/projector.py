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

def test_projector():
    # whisper dim 1024
    input_dim = 1024
    # llama dim 4096
    output_dim = 4096
    
    projector = AudioProjector(input_dim, output_dim)
    
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    output = projector(dummy_input)
    
    # check output shape
    assert output.shape == (batch_size, seq_len, output_dim)
    # check parameters are trainable
    for param in projector.parameters():
        assert param.requires_grad
    
    print("Projector implementation successful")
