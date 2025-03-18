import json
import subprocess
import os
import random
from tqdm import tqdm
import time
import re
import argparse
import importlib.util

def check_cuda_availability():
    """Check if CUDA is available through PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_transformers_availability():
    """Check if the transformers library is available."""
    return importlib.util.find_spec("transformers") is not None

def run_llama_prompt_mlx(prompt, max_tokens=256, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    """Run a prompt through mlx_lm.generate and return the result (Apple Silicon optimized)."""
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
        print(f"Error running Llama with MLX: {e}")
        print(f"stderr: {e.stderr}")
        return None

def run_llama_prompt_cuda(prompt, max_tokens=256, model_id="meta-llama/Llama-3.2-3B-instruct", device="cuda"):
    """Run a prompt through HuggingFace Transformers on CUDA and return the result."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading model {model_id} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use half precision for efficiency
            device_map=device
        )
        
        # Properly format the prompt based on model architecture
        if "llama" in model_id.lower():
            # Format for Llama models
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            # Generic format for other models
            formatted_prompt = prompt
        
        # Tokenize and generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )
        
        # Decode and return only the new tokens
        response = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        print(f"Error running Llama with CUDA: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_llama_prompt(prompt, max_tokens=256, model=None, device=None):
    """
    Run a prompt through an LLM and return the result.
    
    This function will automatically select the appropriate backend:
    - CUDA if available and requested
    - MLX otherwise
    
    Args:
        prompt: The prompt text
        max_tokens: Maximum number of tokens to generate
        model: Model ID (will use defaults if None)
        device: "cuda" or "mlx" (will auto-detect if None)
    """
    # Auto-detect device if not specified
    if device is None:
        if check_cuda_availability() and check_transformers_availability():
            device = "cuda"
            print("CUDA detected. Using CUDA backend.")
        else:
            device = "mlx"
            print("CUDA not detected or transformers not installed. Using MLX backend.")
    
    # Set default models based on device
    if model is None:
        if device == "cuda":
            model = "meta-llama/Llama-3.2-3B-instruct"
        else:
            model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    # Route to appropriate function
    if device == "cuda":
        return run_llama_prompt_cuda(prompt, max_tokens, model, device)
    else:
        return run_llama_prompt_mlx(prompt, max_tokens, model)

def clean_transcript(text):
    """Clean transcription text to a more natural format."""
    if text.isupper():
        # Convert to sentence case (first word capitalized, rest lowercase)
        sentences = re.split(r'(?<=[.!?])\s+', text.lower())
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        return ' '.join(sentences)
    return text

def generate_audio_instruction_examples(metadata_path, output_path, subset_filter=None, limit=None, delay=1.0):
    """
    Generate instruction examples for audio transcription tasks formatted for AudioLLM dataset.
    
    Args:
        metadata_path: Path to the metadata JSON file
        output_path: Path to save the results
        subset_filter: Optional list of subsets to include (e.g., ['train-clean-100'])
        limit: Optional limit to process only a subset of entries
        delay: Delay between API calls in seconds
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Filter by subset if specified
    if subset_filter:
        metadata = [entry for entry in metadata if entry.get('subset') in subset_filter]
        print(f"Filtered to {len(metadata)} entries from subsets: {', '.join(subset_filter)}")
    
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
    
    # Add more complex instruction templates
    advanced_templates = [
        "Listen to this audio clip and write down the spoken content word for word.",
        "This is a segment from the LibriSpeech dataset. What is being said?",
        "Please create a verbatim transcript of this audio recording.",
        "I need a precise transcription of this speech sample. What is said?",
        "What is the exact text being narrated in this audio?",
        "Provide a detailed transcription of the speech in this audio file.",
        "I'm collecting transcripts of spoken text. What does the speaker say here?",
        "Please listen carefully and transcribe every word in this recording.",
        "Transcribe this audio sample with proper capitalization and punctuation.",
        "What text is being read aloud in this audio segment?"
    ]
    
    instruction_templates.extend(advanced_templates)
    
    examples = []
    
    print(f"Generating instruction examples from {len(metadata)} entries...")
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
                "original_file_id": entry["file_id"],
                "subset": entry.get("subset", "unknown"),
                "speaker_id": entry.get("speaker_id", "unknown")
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
        for i, example in enumerate(examples, 1):
            if i <= 50:  # Only show first 50 examples in text file to avoid huge files
                f.write(f"EXAMPLE {i}\n")
                f.write(f"File: {example.get('file_id', '')}\n")
                f.write(f"Subset: {example.get('metadata', {}).get('subset', 'unknown')}\n")
                f.write(f"Audio: {example['audio_paths']}\n")
                f.write(f"Text (instruction): {example['text']}\n")
                f.write(f"Response: {example['response']}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        if len(examples) > 50:
            f.write(f"... and {len(examples) - 50} more examples (not shown for brevity)")
    
    print(f"Preview of first 50 examples saved to {text_output_path}")
    
    # Create a statistics summary file
    stats_output_path = os.path.splitext(output_path)[0] + "_stats.txt"
    
    # Collect stats
    subsets = {}
    total_duration = 0  # If you had duration info
    total_words = 0
    speaker_counts = {}
    
    for example in examples:
        subset = example.get('metadata', {}).get('subset', 'unknown')
        speaker = example.get('metadata', {}).get('speaker_id', 'unknown')
        
        # Count by subset
        subsets[subset] = subsets.get(subset, 0) + 1
        
        # Count by speaker
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # Count words
        word_count = len(example['response'].split())
        total_words += word_count
    
    # Write stats
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        f.write(f"DATASET STATISTICS\n")
        f.write(f"Total examples: {len(examples)}\n")
        f.write(f"Total words: {total_words}\n")
        f.write(f"Average words per example: {total_words / len(examples):.1f}\n\n")
        
        f.write(f"SUBSET DISTRIBUTION:\n")
        for subset, count in sorted(subsets.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(examples)) * 100
            f.write(f"  - {subset}: {count} examples ({percentage:.1f}%)\n")
        
        f.write(f"\nSPEAKER DISTRIBUTION:\n")
        f.write(f"  Total unique speakers: {len(speaker_counts)}\n")
        top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        f.write(f"  Top 10 speakers:\n")
        for speaker, count in top_speakers:
            percentage = (count / len(examples)) * 100
            f.write(f"    - Speaker {speaker}: {count} examples ({percentage:.1f}%)\n")
    
    print(f"Statistics summary saved to {stats_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate LibriSpeech instruction examples')
    parser.add_argument('--metadata-path', default='librispeech_metadata.json',
                        help='Path to the metadata JSON file')
    parser.add_argument('--output-path', default='audio_instruction_examples.json',
                        help='Path to save the generated examples')
    parser.add_argument('--subsets', nargs='+', 
                        help='Only include specific LibriSpeech subsets')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of examples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', choices=['cuda', 'mlx', 'auto'], default='auto',
                        help='Device to use for inference (cuda, mlx, or auto)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model ID to use for generation')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Check if metadata file exists
    if not os.path.exists(args.metadata_path):
        print(f"Metadata file not found: {args.metadata_path}")
        return
    
    # Determine device to use
    device = args.device
    if device == 'auto':
        if check_cuda_availability() and check_transformers_availability():
            device = 'cuda'
        else:
            device = 'mlx'
    
    print(f"Selected device: {device}")
    if args.model:
        print(f"Using model: {args.model}")
    
    # Generate training examples
    generate_audio_instruction_examples(
        args.metadata_path, 
        args.output_path, 
        subset_filter=args.subsets,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
