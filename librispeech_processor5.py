import json
import subprocess
import os
import random
from tqdm import tqdm
import time
import re

def run_llama_prompt(prompt, max_tokens=256, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    """Run a prompt through mlx_lm.generate and return the result."""
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--max-tokens", str(max_tokens),
        "--prompt", prompt
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract just the generated text, removing metadata and status info
        output = result.stdout.strip()
        
        # Remove the "==========" lines and everything before/after them
        if "==========" in output:
            parts = output.split("==========")
            if len(parts) >= 3:  # Should have at least 3 parts if there are two delimiter lines
                return parts[1].strip()
        
        # If we didn't find the expected format, just return the raw output
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error running Llama: {e}")
        print(f"stderr: {e.stderr}")
        return None

def clean_transcript(text):
    """Clean transcription text to a more natural format."""
    if text.isupper():
        # Convert to sentence case (first word capitalized, rest lowercase)
        sentences = re.split(r'(?<=[.!?])\s+', text.lower())
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        return ' '.join(sentences)
    return text

# Removed format_audio_instruction as the dataset handles audio tokens

def generate_audio_instruction_examples(metadata_path, output_path, limit=None, delay=1.0):
    """
    Generate instruction examples for audio transcription tasks formatted for AudioLLM dataset.
    
    Args:
        metadata_path: Path to the metadata JSON file
        output_path: Path to save the results
        limit: Optional limit to process only a subset of entries
        delay: Delay between API calls in seconds
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Limit the number of entries if specified
    if limit and limit < len(metadata):
        # Random sampling might give better diversity than just taking the first N
        metadata = random.sample(metadata, limit)
    
    # Define instruction templates for audio transcription
    instruction_templates = [
        "What does the person say in this audio clip?",
        "Please transcribe this audio recording.",
        "Can you tell me what was said in this audio?",
        "Convert this speech to text.",
        "Write down what you hear in this audio clip.",
        "What words were spoken in this recording?",
        "Provide a transcription of this audio.",
        "What is being said in this audio file?",
        "Please put into text what is spoken in this audio.",
        "Transcribe the spoken content of this recording."
    ]
    
    examples = []
    
    print("Generating instruction examples...")
    for entry in tqdm(metadata):
        transcript = entry["transcription"]
        audio_path = entry["audio_path"]
        
        # Clean the transcript to be more natural
        cleaned_transcript = clean_transcript(transcript)
        
        # Select a random instruction template
        instruction = random.choice(instruction_templates)
        
        # Create the example in the format expected by AudioLLMDataset
        example = {
            "file_id": entry["file_id"],  # Keep this for reference
            "audio_paths": audio_path,    # Changed from "audio_path" to "audio_paths" as expected by dataset
            "text": instruction,          # This is what dataset.py expects as the instruction/prompt
            "response": cleaned_transcript, # Keep the response as is
            "metadata": {                  # Additional metadata that might be useful
                "original_file_id": entry["file_id"]
            }
        }
        
        examples.append(example)
    
    # Save the examples
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(examples)} training examples, saved to {output_path}")
    
    # Create a formatted text file for easier reviewing
    text_output_path = os.path.splitext(output_path)[0] + ".txt"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples):
            f.write(f"EXAMPLE {i+1}\n")
            f.write(f"File: {example.get('file_id', '')}\n")
            f.write(f"Audio: {example['audio_paths']}\n")
            f.write(f"Text (instruction): {example['text']}\n")
            f.write(f"Response: {example['response']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"Formatted version saved to {text_output_path}")

def main():
    metadata_path = "librispeech_test_clean_metadata.json"
    output_path = "audio_instruction_examples.json"
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Generate training examples
    # Adjust limit based on how many examples you need
    generate_audio_instruction_examples(metadata_path, output_path, limit=100)

if __name__ == "__main__":
    main()
