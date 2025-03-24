import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import concurrent.futures
import soundfile as sf
import numpy as np
from itertools import chain

def process_audio(args):
    """Process a single audio file"""
    subset, idx, item, output_dir = args
    audio_data = item['audio']
    sampling_rate = audio_data['sampling_rate']
    
    audio_filename = f"{subset}_{idx:06d}.wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    sf.write(audio_path, audio_data['array'], sampling_rate)
    
    return {
        'id': f"{subset}_{idx}",
        'audio_filename': audio_filename,
        'duration': item['duration'],
        'text': item['text'],
        'text_norm': item['text_norm'],
        'whisper_transcript': item['whisper_transcript'],
        'whisper_transcript_norm': item['whisper_transcript_norm'],
        'wer': item['wer'],
        'sampling_rate': sampling_rate
    }

def process_subset(args):
    """Process an entire subset in parallel"""
    dataset_name, subset, split, max_wer, output_dir, chunk_size = args
    
    print(f"\nProcessing subset: {subset}")
    print(f"Loading dataset {dataset_name} ({subset})...")
    
    # Load the dataset
    dataset = load_dataset(
        dataset_name,
        subset,
        split=split,
        trust_remote_code=True
    )
    
    # Filter by WER if needed
    if max_wer is not None:
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: x['wer'] <= max_wer)
        print(f"Filtered {original_size - len(dataset)} samples with WER > {max_wer}%")
        print(f"Remaining samples: {len(dataset)}")
    
    # Prepare arguments for process_audio
    process_args = [(subset, idx, item, output_dir) for idx, item in enumerate(dataset)]
    
    # Process in chunks to optimize memory usage
    results = []
    num_chunks = (len(process_args) + chunk_size - 1) // chunk_size
    
    # Process chunks in parallel using a local process pool
    with Pool(processes=max(1, min(8, cpu_count() // 2))) as pool:
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(process_args))
            chunk_args = process_args[start_idx:end_idx]
            
            chunk_results = list(tqdm(
                pool.imap(process_audio, chunk_args),
                total=len(chunk_args),
                desc=f"Processing {subset} chunk {i+1}/{num_chunks}"
            ))
            
            results.extend(chunk_results)
    
    return results

def download_huggingface_dataset(
    dataset_name="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
    subsets=None,
    output_dir="./data/huggingface",
    output_metadata="metadata.json",
    split="train",
    max_wer=None,
    max_workers=None,
    chunk_size=100
):
    """
    Download and format the Hugging Face dataset for use with AudioLLM.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        subsets: List of subsets to use
        output_dir: Directory to save processed data
        output_metadata: Name of the metadata file to save
        split: Dataset split to use (train/test/validation)
        max_wer: Maximum WER to include
        max_workers: Maximum number of worker processes for subset processing
        chunk_size: Number of audio files to process in each chunk
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if subsets is None:
        subsets = ["en-ls", "en-gigaspeech-l"]  # Default to these two subsets
    
    # Determine max workers based on available CPUs and number of subsets
    if max_workers is None:
        max_workers = min(len(subsets), max(1, cpu_count() - 1))
    
    # Process subsets in parallel using ThreadPoolExecutor (for I/O bound subset loading)
    # and ProcessPoolExecutor (for CPU bound audio processing)
    all_metadata = []
    subset_args = [(dataset_name, subset, split, max_wer, output_dir, chunk_size) for subset in subsets]
    
    print(f"Processing {len(subsets)} subsets using {max_workers} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_subset, args) for args in subset_args]
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing subsets"):
            try:
                subset_metadata = future.result()
                all_metadata.extend(subset_metadata)
            except Exception as e:
                print(f"Error processing subset: {e}")
    
    # Generate instruction examples in parallel
    examples = generate_instruction_examples(all_metadata)
    
    # Save both metadata and examples
    metadata_path = os.path.join(output_dir, output_metadata)
    examples_path = os.path.join(output_dir, 'instruction_examples.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
        
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    print(f"Instruction examples saved to {examples_path}")
    
    print(f"Dataset processed and saved to {output_dir}")
    print(f"Total samples: {len(all_metadata)}")
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
    
    # Process in parallel chunks for faster generation
    with Pool(processes=max(1, cpu_count() // 2)) as pool:
        chunk_size = 1000
        for i in range(0, len(metadata), chunk_size):
            chunk = metadata[i:i+chunk_size]
            
            # Create instruction examples in parallel
            results = list(tqdm(
                pool.map(
                    lambda entry: {
                        "id": entry["id"],
                        "audio_filename": entry["audio_filename"],
                        "instruction": random.choice(instruction_templates),
                        "input": "",
                        "output": entry["text"],
                        "whisper_output": entry["whisper_transcript"],
                        "wer": entry["wer"]
                    },
                    chunk
                ),
                total=len(chunk),
                desc=f"Generating examples {i}-{i+len(chunk)}/{len(metadata)}"
            ))
            
            examples.extend(results)
    
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
    parser.add_argument("--max-workers", type=int, default=None,
                      help="Maximum number of worker processes for parallel subset processing")
    parser.add_argument("--chunk-size", type=int, default=100,
                      help="Number of audio files to process in each chunk")
    
    args = parser.parse_args()
    
    download_huggingface_dataset(
        dataset_name=args.dataset,
        subsets=args.subsets,
        max_wer=args.max_wer,
        output_dir=args.output_dir,
        output_metadata=args.metadata,
        split=args.split,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size
    )