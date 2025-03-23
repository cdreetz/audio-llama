# models/audio_llm.py
import torch
import torch.nn as nn
from models.base import load_base_models
from models.projector import AudioProjector
from models.lora import apply_lora_to_llama, lora_forward_hook

class AudioLLM(nn.Module):
    def __init__(self, llama_path, whisper_path, lora_rank=320):
        super().__init__()
        
        # load base models
        self.llama, self.whisper_encoder = load_base_models(llama_path, whisper_path)
        
        # Create projector (from Whisper dim to LLaMA dim)
        #whisper_dim = self.whisper_encoder.model.config.d_model  # e.g., 1024
        features_dim = 128
        llama_dim = self.llama.model.config.hidden_size  # e.g., 4096
        print(f"Whisper dimension: {features_dim}, LLaMA dimension: {llama_dim}")

        self.projector = AudioProjector(features_dim, llama_dim)
        
        # apply LoRA to LLaMA
        self.lora_layers = apply_lora_to_llama(self.llama.model, rank=lora_rank)
        
        # register forward hooks to apply LoRA
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

        # temporal subsampling convs
        self.conv1 = nn.Conv1d(80, 512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(512, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()


    
    def forward(self, input_ids=None, attention_mask=None, audio_features=None, labels=None, **kwargs):
        """
        forward pass that handles both text and audio inputs

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask for text [batch_size, seq_len]
            audio_features: Processed audio features [batch_size, seq_len, audio_seq_len, feat_dim]
            labels: Target output IDs for training [batch_size, seq_len]

        """
        device = input_ids.device

        if next(self.llama.model.parameters()).device != device:
            self.llama.model = self.llama.model.to(device)

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
                #audio_features.shape[2],
                audio_seq_len=128
            )

            if labels is not None:
                audio_embed_len = combined_embeddings.shape[1] - text_embeddings.shape[1]
                print(f"Audio embedding length: {audio_embed_len}, Text embedding length: {text_embeddings.shape[1]}")

                batch_size = labels.shape[0]
                audio_padding = torch.full((batch_size, audio_embed_len), -100, device=labels.device)
                adjusted_labels = torch.cat([audio_padding, labels], dim=1)
                print(f"Combined embedding shape: {combined_embeddings.shape}")
                print(f"Adjusted labels shape: {adjusted_labels.shape}")
            else:
                adjusted_labels = labels

        else:
            combined_embeddings = text_embeddings
            combined_attention_mask = attention_mask
            adjusted_labels = labels


        outputs = self.llama.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=adjusted_labels,
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
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Audio features shape: {audio_features.shape}")
        batch_size, text_seq_len, hidden_dim = text_embeddings.shape
        device = text_embeddings.device

        if audio_features is None:
            return text_embeddings

        processed_audio = self._process_audio_features(audio_features)
        projected_audio = self.projector(processed_audio)

        # add audio start/end token embeddings
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.audio_start_token)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.audio_end_token)

        vocab_size = self.llama.model.model.embed_tokens.weight.shape[0]
        print(f"Audio start token ID: {audio_start_id}, Audio end token ID: {audio_end_id}, Vocab size: {vocab_size}")

        # Make sure the IDs are valid
        if audio_start_id >= vocab_size or audio_end_id >= vocab_size:
            raise ValueError(f"Token IDs {audio_start_id}, {audio_end_id} are outside vocabulary size {vocab_size}")

        audio_start_tokens = torch.tensor([[audio_start_id]] * batch_size, device=device)
        audio_end_tokens = torch.tensor([[audio_end_id]] * batch_size, device=device)

        # get embeddings for audio delimiter tokens
        # create batch of single tokens with shape [batch_size, 1]
        audio_start_embedding = self.llama.model.model.embed_tokens(audio_start_tokens)
        audio_end_embedding = self.llama.model.model.embed_tokens(audio_end_tokens)

        audio_start_embedding = audio_start_embedding.to(device)
        projected_audio = projected_audio.to(device)
        audio_end_embedding = audio_end_embedding.to(device)
        text_embeddings = text_embeddings.to(device)

        # concatenate: <audio> + audio_embeddings + </audio> + text_embeddings
        #combined_embeddings = torch.cat([
        #    audio_start_embedding,
        #    projected_audio,
        #    audio_end_embedding,
        #    text_embeddings
        #], dim=1)

        #combined_embeddings = torch.cat([projected_audio, text_embeddings], dim=1)
        combined_embeddings = torch.cat([
            audio_start_embedding,
            projected_audio,
            audio_end_embedding,
            text_embeddings
        ], dim=1)

        print(f"Final combined shape: {combined_embeddings.shape}")

        return combined_embeddings

    def _extend_attention_mask(self, attention_mask, audio_seq_len, has_special_tokens=True):
        """
        extends the attention mask to include prepended audio tokens

        Args:
            attention_mask: [batch_size, text_seq_len]
            audio_seq_len: Number of audio tokens
        """
        batch_size, text_seq_len = attention_mask.shape
        device = attention_mask.device

        if has_special_tokens:
            total_audio_len = audio_seq_len + 2
        else:
            total_audio_len = audio_seq_len

        audio_attention = torch.ones(batch_size, total_audio_len, device=device)

        extended_mask = torch.cat([audio_attention, attention_mask], dim=1)

        return extended_mask

    def _process_audio_features(self, audio_features):
        """
        process audio features with proper temporal subsampling.
        reduces temporal dimension by 8x

        Args:
            audio_features: [batch_size, channels, features, time]

        Returns:
            process_features: [batch_size, time/8, features]
        """
        device = audio_features.device

        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        self.conv3 = self.conv3.to(device)

        if len(audio_features.shape) == 3:
            raise ValueError("Raw audio detected. Dataset should be outputting mel spectograms")
        elif len(audio_features.shape) == 4:
            batch_size, channels, features, time = audio_features.shape
            x = audio_features.squeeze(1)
        else:
            raise ValueError("Unexpected audio features shape: {audio_features.shape}")

        #if len(audio_features.shape) == 3:
        #    batch_size, channels, time = audio_features.shape
        #    x = audio_features.view(batch_size, channels, -1, time // channels)
        #    x = x.squeeze(1).permute(0, 2, 1)
        #elif len(audio_features.shape) == 4:
        #    batch_size, channels, features, time = audio_features.shape
        #    x = audio_features.squeeze(1).permute(0, 2, 1)
        #else:
        #    raise ValueError(f"Unexpected audio features shape: {audio_features.shape}")


        # apply convolutions (need to switch dimension for conv1d)
        #x = x.transpose(1, 2) # [batch, features, time]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2) # [batch, time/8, features]

        return x

    
    def get_trainable_params(self):
        """Return only trainable parameters (projector + LoRA)"""
        params = list(self.projector.parameters())
        for lora in self.lora_layers.values():
            params.extend(list(lora.parameters()))
        return params

    def to(self, device):
        self.llama.to(device)
        self.whisper_encoder.to(device)

        self.projector = self.projector.to(device)

        for layer_name in self.lora_layers:
            self.lora_layers[layer_name] = self.lora_layers[layer_name].to(device)

        return super().to(device)


    def generate(self, 
                input_ids=None, 
                attention_mask=None, 
                audio_features=None, 
                max_new_tokens=256, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True,
                **kwargs):
        """
        Generate text with optional audio context
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask for text [batch_size, seq_len]
            audio_features: Processed audio features [batch_size, channels, features, time]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters passed to LLaMA
            
        Returns:
            generated_text: The generated text as a string
        """
        self.eval()
        device = input_ids.device
        
        # Get the initial text embeddings
        text_embeddings = self.llama.model.model.embed_tokens(input_ids)
        
        # Combine with audio if provided
        if audio_features is not None:
            combined_embeddings = self._combine_text_and_audio_embeddings(
                text_embeddings,
                audio_features,
                input_ids
            )
            
            # Create extended attention mask that includes audio tokens
            batch_size, text_seq_len = attention_mask.shape
            audio_seq_len = combined_embeddings.shape[1] - text_embeddings.shape[1]
            audio_attention = torch.ones(batch_size, audio_seq_len, device=device)
            combined_attention_mask = torch.cat([audio_attention, attention_mask], dim=1)
        else:
            combined_embeddings = text_embeddings
            combined_attention_mask = attention_mask
        
        # Set LLaMA's generation parameters
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id if hasattr(self, 'tokenizer') and self.tokenizer is not None else None,
            "bos_token_id": self.tokenizer.bos_token_id if hasattr(self, 'tokenizer') and self.tokenizer is not None else None,
            "eos_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') and self.tokenizer is not None else None,
        }
        
        # Add any additional kwargs
        generation_config.update(kwargs)
        
        # Generate tokens
        with torch.no_grad():
            outputs = self.llama.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **generation_config
            )
        
        # Calculate where the actual generated content starts
        input_length = input_ids.shape[1]
        if audio_features is not None:
            input_length += audio_seq_len
        
        # Get only the newly generated tokens
        generated_tokens = outputs[0, input_length:]
        
        # Decode to text
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            # If tokenizer isn't available in the model, return the token IDs
            generated_text = generated_tokens
        
        return generated_text

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
