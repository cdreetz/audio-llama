# models/audio_llm.py
import torch
import torch.nn as nn
from models.base import load_base_models
from models.projector import AudioProjector
from models.lora import apply_lora_to_llama, lora_forward_hook

class AudioLLM(nn.Module):
    def __init__(self, llama_path, whisper_path, lora_rank=320):
        super().__init__()
        
        # Load base models
        self.llama, self.whisper_encoder = load_base_models(llama_path, whisper_path)
        
        # Create projector (from Whisper dim to LLaMA dim)
        whisper_dim = self.whisper_encoder.model.config.d_model  # e.g., 1024
        llama_dim = self.llama.model.config.hidden_size  # e.g., 4096
        self.projector = AudioProjector(whisper_dim, llama_dim)
        
        # Apply LoRA to LLaMA
        self.lora_layers = apply_lora_to_llama(self.llama.model, rank=lora_rank)
        
        # Register forward hooks to apply LoRA
        self.hooks = []
        for name, module in self.llama.model.named_modules():
            if name in self.lora_layers:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: lora_forward_hook(mod, inp, out, self.lora_layers[n])
                )
                self.hooks.append(hook)

        self.audio_start_token = "<audio>"
        self.audio_end_token = "</audio>"

        self.tokenizer = None
    
    def forward(self, input_ids=None, attention_mask=None, audio_features=None, labels=None, **kwargs):
        """
        forward pass that handles both text and audio inputs

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask for text [batch_size, seq_len]
            audio_features: Processed audio features [batch_size, seq_len, audio_seq_len, feat_dim]
            labels: Target output IDs for training [batch_size, seq_len]
        """
        # 1. get embeddings from llama model directly
        text_embeddings = self.llama.model.model.embed_tokens(input_ids)

        # 2. process audio through whisper encoder
        if audio_features is not None:
            combined_embeddings = self._combine_text_and_audio_embeddings(
                text_embeddings,
                audio_features,
                input_ids
            )

            combined_attention_mask = self._extend_attention_mask(
                attention_mask,
                audio_features.shape[1],
            )

        else:
            combined_embeddings = text_embeddings
            combined_attention_mask = attention_mask

        # adjust labels if needed as they may need to account for audio tokens
        if labels is not None and audio_features is not None:
            # shift labels or use a loss mask to ignore predictions for audio tokens
            pass

        outputs = self.llama.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs


    def _combine_text_and_audio_embeddings(self, text_embeddings, audio_features, input_ids):
        """
        combines text and audio embeddings by replacing audio placeholder tokens

        Args:
            text_embeddings: [batch_size, text_seq_len, hidden_dim]
            audio_features: [batch_size, text_seq_len, hidden_dim] or None
            input_ids: [batch_size, text_seq_len]

        Returns:
            combined_embeddings: [batch_size, combined_seq_len, hidden_dim]
        """
        batch_size, text_seq_len, hidden_dim = text_embeddings.shape
        device = text_embeddings.device

        if audio_features is None:
            return text_embeddings

        # process audio features through projector
        projected_audio = self.projector(audio_features)

        # add audio start/end token embeddings
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.audio_start_token)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.audio_end_token)

        audio_start_tokens = torch.tensor([[audio_start_id]] * batch_size, device=device)
        audio_end_tokens = torch.tensor([[audio_end_id]] * batch_size, device=device)

        # get embeddings for audio delimiter tokens
        # create batch of single tokens with shape [batch_size, 1]
        audio_start_embedding = self.llama.model.model.embed_tokens(audio_start_tokens)
        audio_end_embedding = self.llama.model.model.embed_tokens(audio_end_tokens)


        print(f"audio_start_embedding shape: {audio_start_embedding.shape}")
        print(f"projected_audio shape: {projected_audio.shape}")
        print(f"audio_end_embedding shape: {audio_end_embedding.shape}")
        print(f"text_embedding shape: {text_embeddings.shape}")


        # concatenate: <audio> + audio_embeddings + </audio> + text_embeddings
        combined_embeddings = torch.cat([
            audio_start_embedding,
            projected_audio,
            audio_end_embedding,
            text_embeddings
        ], dim=1)

        print(f"Final combined shape: {combined_embeddings.shape}")

        return combined_embeddings

    def _extend_attention_mask(self, attention_mask, audio_seq_len):
        """
        extends the attention mask to include prepended audio tokens

        Args:
            attention_mask: [batch_size, text_seq_len]
            audio_seq_len: Number of audio tokens
        """
        batch_size, text_seq_len = attention_mask.shape
        device = attention_mask.device

        audio_attention = torch.ones(batch_size, audio_seq_len + 2, device=device)

        extended_mask = torch.cat([audio_attention, attention_mask], dim=1)

        return extended_mask

    
    def get_trainable_params(self):
        """Return only trainable parameters (projector + LoRA)"""
        params = list(self.projector.parameters())
        for lora in self.lora_layers.values():
            params.extend(list(lora.parameters()))
        return params

# Test function for integrated model
def test_integration():
    model = AudioLLM("llama-7b", "whisper-large-v2")
    
    # Check only projector and LoRA params are trainable
    for name, param in model.named_parameters():
        if "projector" in name or "lora" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad
    
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Model integration successful")
