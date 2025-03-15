# models/base.py
import torch
from transformers import LlamaForCausalLM, WhisperModel

class FrozenModelWrapper:
    def __init__(self, model):
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*args, **kwargs)

def load_base_models(llama_model_path, whisper_model_path):
    llama = LlamaForCausalLM.from_pretrained(llama_model_path)
    whisper_encoder = WhisperModel.from_pretrained(whisper_model_path).encoder
    
    frozen_llama = FrozenModelWrapper(llama)
    frozen_whisper = FrozenModelWrapper(whisper_encoder)
    
    return frozen_llama, frozen_whisper

def test_base_models():
    llama, whisper = load_base_models("llama-7b", "whisper-large-v2")
    # Check models loaded correctly
    assert llama is not None
    assert whisper is not None
    # Check models are frozen
    for param in llama.model.parameters():
        assert not param.requires_grad
    for param in whisper.model.parameters():
        assert not param.requires_grad
    print("Base models loaded and frozen correctly")
