import os
import json
import torch
import random
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, LlamaTokenizer, AutoTokenizer
from dataset import AudioLLMDataset, collate_fn


def create_dataloaders(
    metadata_path,
    audio_dir="./data/huggingface",
    whisper_model_id="openai/whisper-large-v3-turbo",
    llama_model_id="meta-llama/Llama-3.2-3B-Instruct",
    batch_size=8,
    train_split=0.9,
    max_audio_length=30,
    text_max_length=512,
    num_workers=4,
    seed=42,
    skip_missing_files=False,
    use_dummy_audio_for_missing=True
):
    """
    create train and val dataloaders for AudioLLM training

    Args:
        data_path (str): path to json file containing dataset entries
        audio_dir (str): dir containing audio files
        whisper_model_id (str): name of whisper model for processor
        llama_model_id (str): name of llama model for tokenizer
        batch_size (int): batch size for training
        train_split (float): split of data for training
        max_audio_length (int): max audio length in seconds
        text_max_length (int): max text length in tokens
        num_workers (int): number of workers for DataLoader
        seed (int): random seed for reproducability
    """

    random.seed(seed)

    # Load processors
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    with open(metadata_path, 'r', encoding='utf-8') as f:
        data_entries = json.load(f)
        # Metadata is already a list from huggingface_dataset.py
    
    random.shuffle(data_entries)
    split_idx = int(len(data_entries) * train_split)
    train_entries = data_entries[:split_idx]
    val_entries = data_entries[split_idx:]

    
    # Create dataset
    train_dataset = AudioLLMDataset(
        train_entries,
        audio_dir=audio_dir,
        whisper_processor=whisper_processor,
        llama_tokenizer=llama_tokenizer,
        max_audio_length=max_audio_length,
        text_max_length=text_max_length,
        skip_missing_files=skip_missing_files,
        use_dummy_audio_for_missing=use_dummy_audio_for_missing
    )

    val_dataset = AudioLLMDataset(
        val_entries,
        audio_dir=audio_dir,
        whisper_processor=whisper_processor,
        llama_tokenizer=llama_tokenizer,
        max_audio_length=max_audio_length,
        text_max_length=text_max_length,
        skip_missing_files=skip_missing_files,
        use_dummy_audio_for_missing=use_dummy_audio_for_missing
    )
    
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

def get_sample_batch(dataloader):
    dataiter = iter(dataloader)
    sample_batch = next(dataiter)

    print("Sample batch structure:")
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {type(value)} with shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    return sample_batch


if __name__ == "__main__":
    # example usage
    data_path = "path/to/data.json"
    audio_dir = "path/to/audio"

    train_dataloader, val_dataloader = create_dataloaders(
        data_path=data_path,
        audio_dir=audio_dir,
        batch_size=4
    )
