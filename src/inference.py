#!/usr/bin/env python
# Audio-LLM Inference Script

import torch
import torchaudio
import argparse
from transformers import LlamaForCausalLM, WhisperModel, AutoTokenizer, AutoProcessor

from models.allm import AudioLLM
from models.projector import AudioProjector
from models.lora import LoRALayer

def load_audio_llm(
    llama_path="meta-llama/Llama-3.2-3B-Instruct",
    whisper_path="openai/whisper-large-v3-turbo",
    checkpoint_path="cdreetz/audio-llama",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load the AudioLLM model with pretrained projector and LoRA weights
    
    Args:
        llama_path: Path or HF name of LLaMA model
        whisper_path: Path or HF name of Whisper model
        checkpoint_path: Path to the trained projector and LoRA weights
        device: Device to load model on
        
    Returns:
        model: Loaded AudioLLM model
        tokenizer: LLaMA tokenizer
        processor: Whisper processor
    """
    print(f"Loading base models from {llama_path} and {whisper_path}...")
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(whisper_path)
    
    # Add special tokens for audio if needed
    special_tokens = {"additional_special_tokens": ["<audio>", "</audio>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize the model
    model = AudioLLM(llama_path=llama_path, whisper_path=whisper_path)
    model.tokenizer = tokenizer 
    
    # Resize token embeddings after adding special tokens
    model.llama.model.resize_token_embeddings(len(tokenizer))
    
    # Load trained projector and LoRA weights
    print(f"Loading trained weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Full checkpoint format
            model.projector.load_state_dict(checkpoint['model']['projector'])
            for name, state_dict in checkpoint['model']['lora_layers'].items():
                if name in model.lora_layers:
                    model.lora_layers[name].load_state_dict(state_dict)
        else:
            # Direct state dict format
            model.projector.load_state_dict(checkpoint['projector'])
            for name, state_dict in checkpoint['lora_layers'].items():
                if name in model.lora_layers:
                    model.lora_layers[name].load_state_dict(state_dict)
                    
        print("Successfully loaded trained weights!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    model.to(device)
    return model, tokenizer, processor


def process_audio(audio_path, processor, max_length=30, sample_rate=16000, device="cuda"):
    """Process audio file and convert to model features"""
    print(f"Processing audio file: {audio_path}")
    
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        
        # Limit to max_length seconds
        max_samples = sample_rate * max_length
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        # Convert to features using Whisper processor
        input_features = processor(
            waveform.squeeze().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(device)
        
        return input_features
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        raise


def generate_response(
    model, 
    tokenizer, 
    processor,
    text_prompt, 
    audio_path=None, 
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate a response using the AudioLLM model
    
    Args:
        model: AudioLLM model
        tokenizer: LLaMA tokenizer
        processor: Whisper processor
        text_prompt: Text prompt for generation
        audio_path: Path to audio file (optional)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to use
        
    Returns:
        generated_text: Generated response
    """
    model.eval()
    
    # Tokenize text input
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Process audio if provided
    audio_features = None
    if audio_path:
        audio_features = process_audio(audio_path, processor, device=device)
        # Add batch dimension if needed
        if len(audio_features.shape) == 3:
            audio_features = audio_features.unsqueeze(0)
    
    # Use the model's generate method
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features if audio_path else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )
        
        # Extract the generated text (model's generate method returns the decoded text)
        generated_text = outputs
        
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioLLM Inference")
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path to LLaMA model")
    parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v3-turbo",
                        help="Path to Whisper model")
    parser.add_argument("--checkpoint_path", type=str, default="cdreetz/audio-llama",
                        help="Path to AudioLLM checkpoint")
    parser.add_argument("--text_prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to audio file (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, processor = load_audio_llm(
        args.llama_path,
        args.whisper_path,
        args.checkpoint_path,
        args.device
    )
    
    # Generate response
    response = generate_response(
        model,
        tokenizer,
        processor,
        args.text_prompt,
        args.audio_path,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.device
    )
    
    print("\nGenerated Response:")
    print(response)
