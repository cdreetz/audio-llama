#!/usr/bin/env python
"""
Simple example script for using AudioLLM

Example usage:
python example.py --audio_path "samples/audio.wav" --text_prompt "Describe this sound:"
"""

import torch
import argparse
import torchaudio
from inference import load_audio_llm, process_audio

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AudioLLM Example")
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path to LLaMA model")
    parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v3-turbo",
                        help="Path to Whisper model")
    parser.add_argument("--checkpoint_path", type=str, default="cdreetz/audio-llama",
                        help="Path to AudioLLM weights")
    parser.add_argument("--text_prompt", type=str, default="Describe this sound:",
                        help="Text prompt for generation")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to audio file")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=== AudioLLM Example ===")
    print(f"Loading models on {device}...")
    
    # Load the model
    model, tokenizer, processor = load_audio_llm(
        llama_path=args.llama_path,
        whisper_path=args.whisper_path,
        checkpoint_path=args.checkpoint_path,
        device=device
    )
    
    # Tokenize text input
    inputs = tokenizer(args.text_prompt, return_tensors="pt").to(device)
    
    # Process audio file
    print(f"Processing audio: {args.audio_path}")
    audio_features = process_audio(args.audio_path, processor, device=device)
    if len(audio_features.shape) == 3:
        audio_features = audio_features.unsqueeze(0)
    
    print(f"Generating response...")
    print(f"Text prompt: {args.text_prompt}")
    
    # Generate response using model's generate method
    response = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        audio_features=audio_features,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print("\n=== Generated Response ===")
    print(response)

if __name__ == "__main__":
    main()
