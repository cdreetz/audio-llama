#!/usr/bin/env python3

from huggingface_dataset import download_huggingface_dataset

if __name__ == "__main__":
    # Download and process the dataset
    metadata_path = download_huggingface_dataset(
        dataset_name="bofenghuang/stt-pseudo-labeled-whisper-large-v3-multilingual",
        subsets=None,  # Use all available subsets
        output_dir="./data/huggingface",
        output_metadata="metadata.json",
        split="train",
        max_wer=None  # No WER filtering
    )
    print(f"Dataset downloaded and processed. Metadata saved to: {metadata_path}")
