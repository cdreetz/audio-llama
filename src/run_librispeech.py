#!/usr/bin/env python
"""
Librispeech Processing Pipeline
------------------------------
This script demonstrates how to download and process multiple LibriSpeech subsets
and generate instruction examples for audio transcription tasks.

Usage:
    python run_librispeech_processing.py

Optional arguments:
    --download-only: Only download and process the LibriSpeech data, don't generate examples
    --process-only: Only process metadata into examples, don't download data
    --small: Use smaller subsets for faster processing (test-clean, dev-clean)
    --medium: Use medium-sized subsets (test-clean, dev-clean, train-clean-100)
    --large: Use all available subsets
"""
import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='LibriSpeech Processing Pipeline')
    parser.add_argument('--download-only', action='store_true',
                      help='Only download and process the LibriSpeech data')
    parser.add_argument('--process-only', action='store_true',
                      help="Only process metadata into examples, don't download data")
    parser.add_argument('--small', action='store_true',
                      help='Use smaller subsets (test-clean, dev-clean)')
    parser.add_argument('--medium', action='store_true',
                      help='Use medium-sized subsets (test-clean, dev-clean, train-clean-100)')
    parser.add_argument('--large', action='store_true',
                      help='Use all available subsets')
    parser.add_argument('--example-limit', type=int, default=1000,
                      help='Limit the number of examples to generate')
    parser.add_argument('--output-dir', default='librispeech_data',
                      help='Directory to store all outputs')
    
    args = parser.parse_args()
    
    # Determine which subsets to use
    if args.large:
        subsets = ["test-clean", "test-other", "dev-clean", "dev-other", 
                  "train-clean-100", "train-clean-360", "train-other-500"]
    elif args.medium:
        subsets = ["test-clean", "dev-clean", "train-clean-100"]
    elif args.small:
        subsets = ["test-clean", "dev-clean"]
    else:
        # Default to test-clean only
        subsets = ["test-clean"]
    
    print(f"Using LibriSpeech subsets: {', '.join(subsets)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    downloads_dir = os.path.join(args.output_dir, "downloads")
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Set paths
    metadata_path = os.path.join(args.output_dir, "librispeech_metadata.json")
    examples_path = os.path.join(args.output_dir, "audio_instruction_examples.json")
    
    # Step 1: Download and process LibriSpeech data
    if not args.process_only:
        print("\n===== DOWNLOADING AND PROCESSING LIBRISPEECH DATA =====")
        download_cmd = [
            "python", "download_librispeech.py",
            "--subsets"] + subsets + [
            "--download-dir", downloads_dir,
            "--audio-dir", audio_dir,
            "--metadata-path", metadata_path
        ]
        
        if len(subsets) > 1:
            download_cmd.append("--parallel")
        
        print(f"Running command: {' '.join(download_cmd)}")
        start_time = time.time()
        subprocess.run(download_cmd)
        download_time = time.time() - start_time
        print(f"Download and processing completed in {download_time:.1f} seconds")
    
    # Step 2: Generate instruction examples
    if not args.download_only:
        print("\n===== GENERATING INSTRUCTION EXAMPLES =====")
        process_cmd = [
            "python", "src/librispeech_processor7.py",
            "--metadata-path", metadata_path,
            "--output-path", examples_path,
            "--subsets"] + subsets
        
        if args.example_limit:
            process_cmd.extend(["--limit", str(args.example_limit)])
        
        print(f"Running command: {' '.join(process_cmd)}")
        start_time = time.time()
        subprocess.run(process_cmd)
        process_time = time.time() - start_time
        print(f"Example generation completed in {process_time:.1f} seconds")
    
    print("\n===== PROCESSING COMPLETE =====")
    print(f"All outputs saved to {args.output_dir}")
    
    if not args.download_only:
        print(f"Generated examples saved to {examples_path}")
        print("\nSample commands to use the data:")
        print(f"  - View example statistics: cat {os.path.splitext(examples_path)[0]}_stats.txt")
        print(f"  - Preview examples: head -n 50 {os.path.splitext(examples_path)[0]}.txt")

if __name__ == "__main__":
    main()
