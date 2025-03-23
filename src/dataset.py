import os
import torch
import torchaudio
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

class AudioLLMDataset(Dataset):
    def __init__(
        self, 
        data_entries,
        audio_dir="./audio",
        whisper_processor=None, 
        llama_tokenizer=None,
        max_audio_length=30,  # seconds
        sample_rate=16000,
        audio_padding="max_length",
        text_max_length=512,
        skip_missing_files=False,
        use_dummy_audio_for_missing=True
    ):
        self.data = data_entries
        self.audio_dir = audio_dir
        self.whisper_processor = whisper_processor
        self.llama_tokenizer = llama_tokenizer
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.audio_padding = audio_padding
        self.text_max_length = text_max_length

        self.audio_start_token = "<audio>"
        self.audio_end_token = "</audio>"

        if self.audio_start_token not in llama_tokenizer.get_vocab():
            special_tokens = {
                "additional_special_tokens": [self.audio_start_token, self.audio_end_token]
            }
            llama_tokenizer.add_special_tokens(special_tokens)

        self.skip_missing_files = skip_missing_files
        self.use_dummy_audio_for_missing = use_dummy_audio_for_missing
        self.missing_files = []

        if self.skip_missing_files:
            self._filter_missing_files()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        text = item.get("text", "")
        audio_path = item.get("audio_paths", "")
        
        audio_features = None
        if audio_path:
            full_path = os.path.join(self.audio_dir, audio_path)
            audio_features = self._process_audio(full_path)
        
        tokenized = self.llama_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        )

        response = item.get("response", "")
        response_tokenized = self.llama_tokenizer(
            response,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt"
        )

        
        return {
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "audio_features": audio_features,
            "labels": response_tokenized.input_ids.squeeze(0),
            "text": text,
            "audio_path": audio_path,
        }
    
    def _process_audio(self, audio_path):
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros((1, 80, 1000))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0
        )(waveform)

        log_mel = torch.log(mel_spectrogram + 1e-9)

        
        # audio to Whisper features
        #input_features = self.whisper_processor(
        #    waveform.squeeze().numpy(),
        #    sampling_rate=self.sample_rate,
        #    return_tensors="pt"
        #).input_features
        
        # output shape [1, 80, time_frames]
        return log_mel

    
    def _process_text(self, text):
        tokens = self.llama_tokenizer(
            text,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="pt"
        )
        # Remove batch dimension for dataset
        for k, v in tokens.items():
            tokens[k] = v.squeeze(0)
            
        return tokens

    def _filter_missing_files(self):
        valid_entries = []

        for item in self.data:
            audio_path = item.get("audio_paths", "")
            if not audio_path:
                valid_entries.append(item)
                continue
            full_path = os.path.join(self.audio_dir, audio_path)

            if not os.path.exists(full_path) and audio_path.startswith("audio/"):
                fixed_path = audio_path[6:]
                full_path = os.path.join(self.audio_dir, fixed_path)

                if os.path.exists(full_path):
                    item["audio_paths"] = fixed_path

            if os.path.exists(full_path):
                valid_entries.append(item)
            else:
                self.missing_files.append(audio_path)

        print(f"Filtered out {len(self.data) - len(valid_entries)} entires with missing audio files")
        self.data = valid_entries


def collate_fn(batch):
    audio_features = torch.stack([item["audio_features"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "audio_features": audio_features if len(audio_features) > 0 else None,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": [item.get("metadata", {}) for item in batch]
    }
