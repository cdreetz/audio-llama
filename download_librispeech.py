import os
import json
import tarfile
import shutil
import requests
from tqdm import tqdm
from pathlib import Path
import concurrent.futures

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

def process_librispeech(download_dir, audio_dir, metadata_path):
    """Process LibriSpeech dataset to create metadata and organize audio files."""
    # Create metadata dictionary
    metadata = []
    
    # Process the test-clean directory
    extract_dir = os.path.join(download_dir, "LibriSpeech")
    test_clean_dir = os.path.join(extract_dir, "test-clean")
    
    # Convert flac files to desired format and build metadata
    for speaker_id in os.listdir(test_clean_dir):
        speaker_path = os.path.join(test_clean_dir, speaker_id)
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
                        output_dir = os.path.join(audio_dir, speaker_id, chapter_id)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Copy the audio file
                        src_path = os.path.join(chapter_path, file_name)
                        dst_path = os.path.join(output_dir, file_name)
                        shutil.copy2(src_path, dst_path)
                        
                        # Get duration (this would require additional library, using placeholder)
                        # In a real implementation, you could use librosa or pydub to get actual duration
                        
                        if file_id in transcriptions:
                            # Add to metadata
                            metadata.append({
                                "audio_path": os.path.relpath(dst_path, start=os.path.dirname(audio_dir)),
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "file_id": file_id,
                                "transcription": transcriptions[file_id]
                            })
    
    # Save metadata to JSON file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Created metadata for {len(metadata)} audio files")

def main():
    # URLs and paths
    LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    
    # Create directories
    download_dir = "downloads"
    audio_dir = "audio"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Download file
    tar_path = os.path.join(download_dir, "test-clean.tar.gz")
    if not os.path.exists(tar_path):
        download_file(LIBRISPEECH_URL, tar_path)
    
    # Extract archive
    if not os.path.exists(os.path.join(download_dir, "LibriSpeech", "test-clean")):
        extract_tar(tar_path, download_dir)
    
    # Process dataset and create metadata
    metadata_path = "librispeech_test_clean_metadata.json"
    process_librispeech(download_dir, audio_dir, metadata_path)
    
    print(f"Processing complete! Metadata saved to {metadata_path}")
    print(f"Audio files saved to {audio_dir}")

if __name__ == "__main__":
    main()
