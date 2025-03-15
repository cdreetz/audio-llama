import os
import torch
import argparse
import torchaudio
import numpy as np
from transformers import LlamaTokenizer, WhisperProcessor, TextIteratorStreamer
from threading import Thread
from models.audio_llm import AudioLLM

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with AudioLLM model")
    
    # Model paths
    parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path to pre-trained LLaMA model")
    parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v3-turbo",
                        help="Path to pre-trained Whisper model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to fine-tuned AudioLLM checkpoint")
    
    # Input parameters
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to audio file for transcription")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to include with the audio")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--stream", action="store_true",
                        help="Stream output tokens as they're generated")
    
    return parser.parse_args()

def prepare_audio_features(audio_path, processor, sample_rate=16000):
    """Process audio file into features for the model"""
    print(f"Processing audio file: {audio_path}")
    
    # Load and process audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    
    # Convert to features using Whisper processor
    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features
    
    return input_features

def load_model_and_checkpoint(args, device):
    """Load base model and apply fine-tuned parameters"""
    print("Loading model...")
    
    # Initialize model
    model = AudioLLM(
        llama_path=args.llama_path,
        whisper_path=args.whisper_path,
        lora_rank=32  # Use same rank as training
    )
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
    
    # Add special tokens for audio
    audio_tokens = {"additional_special_tokens": ["<audio>", "</audio>"]}
    tokenizer.add_special_tokens(audio_tokens)
    
    # Set tokenizer in model
    model.tokenizer = tokenizer
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Load projector weights
    model.projector.load_state_dict(checkpoint['model']['projector'])
    
    # Load LoRA weights
    for name, state_dict in checkpoint['model']['lora_layers'].items():
        if name in model.lora_layers:
            model.lora_layers[name].load_state_dict(state_dict)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, input_ids, audio_features, args, device):
    """Generate a response from the model"""
    
    # Setup generation parameters
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.temperature > 0
    }
    
    # Prepare inputs
    inputs = {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones(input_ids.shape, dtype=torch.long, device=device),
    }
    
    if audio_features is not None:
        inputs["audio_features"] = audio_features.to(device)
    
    # Stream output if requested
    if args.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        gen_config["streamer"] = streamer
        
        # Run generation in a separate thread
        thread = Thread(target=model.llama.model.generate, kwargs={**inputs, **gen_config})
        thread.start()
        
        # Print streaming output
        print("\nGenerated response:")
        for text in streamer:
            print(text, end="", flush=True)
        print("\n")
    else:
        # Standard generation
        with torch.no_grad():
            generated_ids = model.llama.model.generate(**inputs, **gen_config)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\nGenerated response:")
        print(generated_text)
        
        return generated_text

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and checkpoint
    model, tokenizer = load_model_and_checkpoint(args, device)
    
    # Load Whisper processor
    whisper_processor = WhisperProcessor.from_pretrained(args.whisper_path)
    
    # Process audio if provided
    audio_features = None
    if args.audio_path and os.path.exists(args.audio_path):
        audio_features = prepare_audio_features(args.audio_path, whisper_processor)
    
    # Prepare text prompt
    if args.audio_path:
        # Add audio placeholder
        prompt = f"{args.prompt}<audio></audio>"
    else:
        prompt = args.prompt
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate response
    generate_response(model, tokenizer, input_ids, audio_features, args, device)

if __name__ == "__main__":
    main()
