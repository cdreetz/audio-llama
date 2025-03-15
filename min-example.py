"""
Minimal end-to-end example to test the AudioLLM model with a small dataset
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, WhisperProcessor, AutoTokenizer

from models.audio_llm import AudioLLM
from dataset import AudioLLMDataset, collate_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    llama_path = "meta-llama/Llama-3.2-3B-Instruct"  
    whisper_path = "openai/whisper-large-v3-turbo"
    
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
    whisper_processor = WhisperProcessor.from_pretrained(whisper_path)
    
    audio_start_token = "<audio>"
    audio_end_token = "</audio>"
    
    if audio_start_token not in llama_tokenizer.get_vocab():
        special_tokens = {
            "additional_special_tokens": [audio_start_token, audio_end_token]
        }
        llama_tokenizer.add_special_tokens(special_tokens)
    
    # Create a small test dataset
    # In a real scenario, you would load this from files
    test_data = [
        {
            "text": "Describe the audio: <audio>",
            "audio_paths": "sample1.wav",  # Make sure this file exists in audio_dir
            "response": "This is a recording of a piano playing."
        },
        {
            "text": "What can you hear in this recording? <audio>",
            "audio_paths": "sample2.wav",  # Make sure this file exists in audio_dir
            "response": "This recording contains a person speaking in English."
        }
    ]
    
    # Create the dataset
    print("Creating dataset...")
    audio_dir = "./audio_samples"  # Directory containing audio files
    
    # Create directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    
    # Mock audio files if they don't exist (just for testing)
    for entry in test_data:
        audio_path = entry.get("audio_paths")
        if audio_path:
            full_path = os.path.join(audio_dir, audio_path)
            if not os.path.exists(full_path):
                print(f"Creating mock audio file: {full_path}")
                # Create a dummy WAV file (1 second of silence)
                import scipy.io.wavfile as wavfile
                import numpy as np
                sample_rate = 16000
                data = np.zeros(sample_rate, dtype=np.int16)
                wavfile.write(full_path, sample_rate, data)
    
    dataset = AudioLLMDataset(
        data_entries=test_data,
        audio_dir=audio_dir,
        whisper_processor=whisper_processor,
        llama_tokenizer=llama_tokenizer,
        text_max_length=512
    )
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize the model
    print("Initializing AudioLLM model...")
    model = AudioLLM(llama_path, whisper_path)
    model.tokenizer = llama_tokenizer  # Set the tokenizer in the model
    
    model.to(device)
    model.eval()
    
    # Process a batch
    print("Processing a batch...")
    batch = next(iter(dataloader))
    
    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audio_features = batch["audio_features"].to(device) if batch["audio_features"] is not None else None
    labels = batch["labels"].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            labels=labels
        )
    
    # Print results
    print("Forward pass successful!")
    print(f"Output loss: {outputs.loss.item()}")
    
    # Generate some text based on the audio
    print("\nGenerating text based on audio input...")
    input_text = "Describe this audio: <audio>"
    
    # Tokenize the input text
    tokenized = llama_tokenizer(input_text, return_tensors="pt").to(device)
    
    # Get the first audio features from the batch as an example
    sample_audio = audio_features[0].unsqueeze(0) if audio_features is not None else None
    
    # Generate text
    with torch.no_grad():
        outputs = model.llama.generate(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            audio_features=sample_audio,
            max_length=100,
            num_beams=4,
            temperature=0.7
        )
    
    # Decode the generated text
    generated_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
