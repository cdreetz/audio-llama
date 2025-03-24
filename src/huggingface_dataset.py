import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import soundfile as sf

def download_huggingface_dataset(
    dataset_name="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
    subsets=None,  # List of subsets to use
    output_dir="./data/huggingface",
    output_metadata="metadata.json",
    split="train",
    max_wer=None  # Maximum WER to include
):
    """
    Download and format the Hugging Face dataset for use with AudioLLM.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subset: Subset of the dataset to use
        output_dir: Directory to save processed data
        output_metadata: Name of the metadata file to save
        split: Dataset split to use (train/test/validation)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if subsets is None:
        subsets = ["en-ls", "en-gigaspeech-l"]  # Default to these two subsets
    
    all_metadata = []
    
    for subset in subsets:
        print(f"\nProcessing subset: {subset}")
        print(f"Loading dataset {dataset_name} ({subset})...")
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            trust_remote_code=True
        )
        
        if max_wer is not None:
            # Filter by WER
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: x['wer'] <= max_wer)
            print(f"Filtered {original_size - len(dataset)} samples with WER > {max_wer}%")
            print(f"Remaining samples: {len(dataset)}")
    
        def process_item(args):
            idx, item = args
            audio_data = item['audio']
            sampling_rate = audio_data['sampling_rate']
            duration = item['duration']
            
            audio_filename = f"{subset}_{idx:06d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # Save audio data
            sf.write(audio_path, audio_data['array'], sampling_rate)
            
            return {
                'id': f"{subset}_{idx}",
                'audio_filename': audio_filename,
                'duration': duration,
                'text': item['text'],
                'text_norm': item['text_norm'],
                'whisper_transcript': item['whisper_transcript'],
                'whisper_transcript_norm': item['whisper_transcript_norm'],
                'wer': item['wer'],
                'sampling_rate': sampling_rate
            }
        
        # Process items in parallel
        print(f"Processing {subset} audio files...")
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
        with Pool(num_workers) as pool:
            subset_metadata = list(tqdm(
                pool.imap(process_item, enumerate(dataset)),
                total=len(dataset)
            ))
        
        all_metadata.extend(subset_metadata)
    
    # Generate instruction examples
    examples = generate_instruction_examples(all_metadata)
    
    # Save both metadata and examples
    metadata_path = os.path.join(output_dir, output_metadata)
    examples_path = os.path.join(output_dir, 'instruction_examples.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    print(f"Instruction examples saved to {examples_path}")
    
    print(f"Dataset processed and saved to {output_dir}")
    print(f"Total samples: {len(metadata)}")
    return metadata_path

def generate_instruction_examples(metadata):
    """
    Generate instruction-response pairs for audio transcription tasks.
    
    Args:
        metadata: List of metadata entries containing transcriptions
    
    Returns:
        List of instruction-response pairs
    """
    # Define instruction templates
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
        "Please create a verbatim transcript of this audio recording.",
        "I need a precise transcription of this speech sample. What is said?",
        "What is the exact text being narrated in this audio?",
        "Provide a detailed transcription of the speech in this audio file.",
        "Please listen carefully and transcribe every word in this recording.",
        "Transcribe this audio sample with proper capitalization and punctuation.",
        "What text is being spoken in this audio segment?"
    ]
    
    instruction_templates.extend(advanced_templates)
    
    examples = []
    
    print("Generating instruction examples...")
    for entry in tqdm(metadata):
        # Create instruction example
        instruction = random.choice(instruction_templates)
        
        example = {
            "id": entry["id"],
            "audio_filename": entry["audio_filename"],
            "instruction": instruction,
            "input": "",  # No additional input needed
            "output": entry["text"],  # Using the original text as ground truth
            "whisper_output": entry["whisper_transcript"],  # Store Whisper's output for reference
            "wer": entry["wer"]  # Store WER for quality assessment
        }
        
        examples.append(example)
    
    return examples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and process Hugging Face dataset")
    parser.add_argument("--dataset", default="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
                      help="Dataset name on Hugging Face")
    parser.add_argument("--subsets", nargs="+", default=["en-ls", "en-gigaspeech-l"],
                      help="Dataset subsets to use (space-separated list)")
    parser.add_argument("--max-wer", type=float, default=5.0,
                      help="Maximum WER (Word Error Rate) to include")
    parser.add_argument("--output-dir", default="./data/huggingface",
                      help="Output directory for processed data")
    parser.add_argument("--metadata", default="metadata.json",
                      help="Name of the metadata file")
    parser.add_argument("--split", default="train",
                      help="Dataset split to use (train/test/validation)")
    
    args = parser.parse_args()
    
    download_huggingface_dataset(
        dataset_name=args.dataset,
        subsets=args.subsets,
        max_wer=args.max_wer,
        output_dir=args.output_dir,
        output_metadata=args.metadata,
        split=args.split
    )
