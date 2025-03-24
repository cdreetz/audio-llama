import os
import json
import tarfile
import shutil
import requests
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import argparse

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

def process_subset(subset_name, download_dir, audio_dir, metadata_list):
    """Process a LibriSpeech subset to create metadata and organize audio files."""
    # Process the subset directory
    extract_dir = os.path.join(download_dir, "LibriSpeech")
    subset_dir = os.path.join(extract_dir, subset_name)
    
    print(f"Processing {subset_name} directory...")
    
    # Convert flac files to desired format and build metadata
    for speaker_id in os.listdir(subset_dir):
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
                            # Add to metadata
                            metadata_list.append({
                                "audio_path": os.path.relpath(dst_path, start=audio_dir),
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "file_id": file_id,
                                "subset": subset_name,
                                "transcription": transcriptions[file_id]
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
    subset_dir = os.path.join(download_dir, "LibriSpeech", subset)
    if not os.path.exists(subset_dir):
        extract_tar(tar_path, download_dir)
    else:
        print(f"Skipping extraction for {subset} (directory already exists)")
    
    # Process subset and return its metadata
    metadata = []
    metadata = process_subset(subset, download_dir, audio_dir, metadata)
    return metadata

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and process LibriSpeech dataset')
    parser.add_argument('--subsets', nargs='+', default=['test-clean'],
                        help='LibriSpeech subsets to download and process')
    parser.add_argument('--download-dir', default='downloads',
                        help='Directory to store downloaded files')
    parser.add_argument('--audio-dir', default='audio',
                        help='Directory to store processed audio files')
    parser.add_argument('--metadata-path', default='librispeech_metadata.json',
                        help='Path to save the metadata JSON file')
    parser.add_argument('--force-download', action='store_true',
                        help='Force download even if files already exist')
    parser.add_argument('--parallel', action='store_true',
                        help='Process subsets in parallel (warning: uses more disk space and bandwidth)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.audio_dir, exist_ok=True)
    
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
    
    # Save metadata to JSON file
    with open(args.metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Metadata saved to {args.metadata_path}")
    print(f"Total entries: {len(all_metadata)}")
    print(f"Audio files saved to {args.audio_dir}")

if __name__ == "__main__":
    main()
