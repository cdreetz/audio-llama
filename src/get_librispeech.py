#!/usr/bin/env python3
"""
LibriSpeech Dataset Downloader

This script downloads and processes LibriSpeech data into a standardized format
compatible with the AudioLLM training pipeline. The script handles:
1. Downloading the requested LibriSpeech subsets
2. Processing audio files into a consistent directory structure
3. Creating metadata and examples files in the standard format

Usage:
    python download_librispeech.py --subsets test-clean dev-clean train-clean-100
    
    For smaller test run:
    python download_librispeech.py --subsets test-clean --limit 100

python get_librispeech.py --subsets test-clean --limit 100 --output_dir data/librispeech_test
"""

import os
import json
import tarfile
import shutil
import requests
import random
import argparse
from tqdm import tqdm
from pathlib import Path
import concurrent.futures

# LibriSpeech dataset subsets and their URLs
LIBRISPEECH_SUBSETS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz"
}

# Standard instruction templates for audio transcription
INSTRUCTION_TEMPLATES = [
    "What does the person say in this audio clip?",
    "Please transcribe this audio recording.",
    "Can you tell me what was said in this audio?",
    "Convert this speech to text.",
    "Write down what you hear in this audio clip.",
    "What words were spoken in this recording?",
    "Provide a transcription of this audio.",
    "What is being said in this audio file?",
    "Please put into text what is spoken in this audio.",
    "Transcribe the spoken content of this recording.",
    "Listen to this audio clip and write down the spoken content word for word.",
    "This is a segment from the LibriSpeech dataset. What is being said?",
    "Please create a verbatim transcript of this audio recording.",
    "I need a precise transcription of this speech sample. What is said?",
    "What is the exact text being narrated in this audio?",
    "Provide a detailed transcription of the speech in this audio file."
]

def download_file(url, output_path):
    """Download a file with progress tracking."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(url)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tar_path, extract_dir):
    """Extract tar file with progress tracking."""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=f"Extracting {os.path.basename(tar_path)}") as bar:
            for member in members:
                tar.extract(member, path=extract_dir)
                bar.update(1)

def clean_text(text):
    """Clean transcription text to a more natural format."""
    # Convert all uppercase text to sentence case
    if text.isupper():
        text = text.lower()
        sentences = text.split('. ')
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        text = '. '.join(sentences)
        
        # Ensure first character is capitalized
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
    
    return text

def process_subset(subset_name, extract_dir, audio_dir, metadata_list, relative_audio_path=True):
    """Process a LibriSpeech subset to create metadata and organize audio files."""
    subset_dir = os.path.join(extract_dir, "LibriSpeech", subset_name)
    
    print(f"Processing {subset_name} directory...")
    
    # Create the subset directory in the audio output directory
    os.makedirs(os.path.join(audio_dir, subset_name), exist_ok=True)
    
    # Convert flac files to desired format and build metadata
    for speaker_id in tqdm(os.listdir(subset_dir), desc=f"Processing speakers in {subset_name}"):
        speaker_path = os.path.join(subset_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
            
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
                
            # Find the transcript file
            transcript_file = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            
            if os.path.exists(transcript_file):
                # Read all transcriptions
                transcriptions = {}
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            transcriptions[file_id] = text
                
                # Process audio files
                for file_name in os.listdir(chapter_path):
                    if file_name.endswith('.flac'):
                        file_id = os.path.splitext(file_name)[0]
                        
                        # Create structure in audio directory
                        output_dir = os.path.join(audio_dir, subset_name, speaker_id, chapter_id)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Copy the audio file
                        src_path = os.path.join(chapter_path, file_name)
                        dst_path = os.path.join(output_dir, file_name)
                        shutil.copy2(src_path, dst_path)
                        
                        if file_id in transcriptions:
                            # Get the relative path for the audio file
                            if relative_audio_path:
                                audio_path = os.path.join(subset_name, speaker_id, chapter_id, file_name)
                            else:
                                audio_path = dst_path
                            
                            # Clean the transcription text
                            clean_transcription = clean_text(transcriptions[file_id])
                            
                            # Add to metadata
                            metadata_list.append({
                                "audio_paths": audio_path,
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "file_id": file_id,
                                "subset": subset_name,
                                "text": "",  # Will be filled with instruction later
                                "response": clean_transcription,
                                "metadata": {
                                    "original_transcript": transcriptions[file_id],
                                    "speaker_id": speaker_id,
                                    "subset": subset_name
                                }
                            })
    
    return metadata_list

def download_and_process_subset(subset, download_dir, audio_dir, force_download=False):
    """Download and process a specific LibriSpeech subset."""
    if subset not in LIBRISPEECH_SUBSETS:
        print(f"Unknown subset: {subset}")
        return []
    
    url = LIBRISPEECH_SUBSETS[subset]
    tar_path = os.path.join(download_dir, f"{subset}.tar.gz")
    
    # Download file if it doesn't exist or force_download is True
    if force_download or not os.path.exists(tar_path):
        download_file(url, tar_path)
    else:
        print(f"Skipping download for {subset} (file already exists)")
    
    # Extract archive if the subset directory doesn't exist
    extract_dir = download_dir
    subset_dir = os.path.join(extract_dir, "LibriSpeech", subset)
    if not os.path.exists(subset_dir):
        extract_tar(tar_path, extract_dir)
    else:
        print(f"Skipping extraction for {subset} (directory already exists)")
    
    # Process subset and return its metadata
    metadata = []
    metadata = process_subset(subset, extract_dir, audio_dir, metadata)
    return metadata

def generate_examples(metadata, output_path, limit=None):
    """
    Generate instruction examples for the dataset.
    Each example consists of an instruction and the corresponding transcription.
    """
    # Shuffle metadata to ensure a good mix if we're limiting
    shuffled_metadata = metadata.copy()
    random.shuffle(shuffled_metadata)
    
    # Limit the number of examples if specified
    if limit and limit < len(shuffled_metadata):
        shuffled_metadata = shuffled_metadata[:limit]
    
    print(f"Generating {len(shuffled_metadata)} examples...")
    
    # Create the examples
    examples = []
    for item in tqdm(shuffled_metadata, desc="Generating examples"):
        # Select a random instruction template
        instruction = random.choice(INSTRUCTION_TEMPLATES)
        
        # Update the text field with the instruction
        item["text"] = instruction
        
        # Add to examples list
        examples.append(item)
    
    # Save the examples
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(examples)} examples, saved to {output_path}")
    return examples

def create_dataset_stats(examples, output_dir):
    """Create dataset statistics summary"""
    stats_output_path = os.path.join(output_dir, "dataset_stats.txt")
    
    # Collect stats
    subsets = {}
    total_words = 0
    speaker_counts = {}
    
    for example in examples:
        subset = example.get('subset', 'unknown')
        speaker = example.get('speaker_id', 'unknown')
        
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
    
    print(f"Statistics saved to {stats_output_path}")

def create_dataset_config(output_dir):
    """Create a dataset configuration file for AudioLLM training"""
    config = {
        "audio_key": "audio_paths",
        "text_key": "text",
        "response_key": "response",
        "dataset_name": "librispeech"
    }
    
    config_path = os.path.join(output_dir, "dataset_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Dataset configuration saved to {config_path}")
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and process LibriSpeech dataset')
    parser.add_argument('--subsets', nargs='+', default=['test-clean'],
                        help='LibriSpeech subsets to download and process')
    parser.add_argument('--download_dir', default='downloads',
                        help='Directory to store downloaded files')
    parser.add_argument('--audio_dir', default='audio',
                        help='Directory to store processed audio files')
    parser.add_argument('--output_dir', default='data/librispeech',
                        help='Directory to save processed data')
    parser.add_argument('--force_download', action='store_true',
                        help='Force download even if files already exist')
    parser.add_argument('--parallel', action='store_true',
                        help='Process subsets in parallel')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of examples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate requested subsets
    valid_subsets = [s for s in args.subsets if s in LIBRISPEECH_SUBSETS]
    invalid_subsets = [s for s in args.subsets if s not in LIBRISPEECH_SUBSETS]
    
    if invalid_subsets:
        print(f"Warning: Unknown subsets will be skipped: {', '.join(invalid_subsets)}")
    
    if not valid_subsets:
        print("No valid subsets specified. Available subsets:")
        for subset in LIBRISPEECH_SUBSETS:
            print(f"  - {subset}")
        return
    
    all_metadata = []
    
    if args.parallel and len(valid_subsets) > 1:
        # Process subsets in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_subset = {
                executor.submit(download_and_process_subset, subset, args.download_dir, args.audio_dir, args.force_download): subset
                for subset in valid_subsets
            }
            
            for future in concurrent.futures.as_completed(future_to_subset):
                subset = future_to_subset[future]
                try:
                    subset_metadata = future.result()
                    all_metadata.extend(subset_metadata)
                    print(f"Completed processing {subset} with {len(subset_metadata)} entries")
                except Exception as e:
                    print(f"Error processing {subset}: {e}")
    else:
        # Process subsets sequentially
        for subset in valid_subsets:
            try:
                subset_metadata = download_and_process_subset(subset, args.download_dir, args.audio_dir, args.force_download)
                all_metadata.extend(subset_metadata)
                print(f"Completed processing {subset} with {len(subset_metadata)} entries")
            except Exception as e:
                print(f"Error processing {subset}: {e}")
    
    # Save raw metadata (without instructions)
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Raw metadata saved to {metadata_path}")
    
    # Generate examples (with instructions)
    examples_path = os.path.join(args.output_dir, "examples.json")
    examples = generate_examples(all_metadata, examples_path, limit=args.limit)
    
    # Create dataset stats
    create_dataset_stats(examples, args.output_dir)
    
    # Create dataset config
    create_dataset_config(args.output_dir)
    
    print(f"Processing complete!")
    print(f"Total examples: {len(examples)}")
    print(f"Audio files saved to {args.audio_dir}")
    print(f"Dataset files saved to {args.output_dir}")
    print("\nTo use this dataset for training, run:")
    print(f"  python train.py --data_path {examples_path} --audio_dir {args.audio_dir} --dataset_config {os.path.join(args.output_dir, 'dataset_config.json')}")

if __name__ == "__main__":
    main()
