import os
import json
import torch
import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import modules to test
from dataset import AudioLLMDataset, collate_fn
from dataloaders import create_dataloaders, get_sample_batch


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that are used across test methods"""
        # Create a temporary directory for our test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.audio_dir = Path(cls.temp_dir.name) / "audio"
        cls.audio_dir.mkdir(exist_ok=True)
        
        # Create dummy data entries
        cls.data_entries = cls._create_dummy_data_entries()
        
        # Write data entries to a JSON file
        cls.data_path = Path(cls.temp_dir.name) / "data.json"
        with open(cls.data_path, 'w', encoding='utf-8') as f:
            json.dump(cls.data_entries, f)
        
        # Mock the tokenizer and processor
        cls.mock_tokenizer = MagicMock()
        cls.mock_tokenizer.get_vocab.return_value = {"<audio>": 1, "</audio>": 2}
        cls.mock_tokenizer.return_value = MagicMock(
            input_ids=torch.tensor([[101, 102, 103]]),
            attention_mask=torch.tensor([[1, 1, 1]])
        )
        
        cls.mock_processor = MagicMock()
        # Mock the return value of the processor to be a tensor of the right shape
        cls.mock_processor.return_value = MagicMock(input_features=torch.randn(1, 80, 3000))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests have run"""
        cls.temp_dir.cleanup()
    
    @classmethod
    def _create_dummy_data_entries(cls):
        """Create dummy data entries for testing"""
        return [
            {
                "text": "This is a test prompt for audio file 0.",
                "audio_paths": "test_audio_0.wav",
                "response": "This is a response for audio file 0."
            },
            {
                "text": "This is a test prompt for audio file 1.",
                "audio_paths": "test_audio_1.wav",
                "response": "This is a response for audio file 1."
            },
            {
                "text": "This is a test prompt with no audio.",
                "audio_paths": "",
                "response": "This is a response for text-only input."
            }
        ]
    
    def test_dataset_initialization(self):
        """Test that the dataset initializes properly"""
        # Patch audio file processing to avoid actual file loading
        with patch.object(AudioLLMDataset, '_process_audio', return_value=torch.randn(1, 80, 3000)):
            # Create dataset
            dataset = AudioLLMDataset(
                data_entries=self.data_entries,
                audio_dir=str(self.audio_dir),
                whisper_processor=self.mock_processor,
                llama_tokenizer=self.mock_tokenizer,
                max_audio_length=30,
                text_max_length=512
            )
            
            # Test length matches input data
            self.assertEqual(len(dataset), len(self.data_entries))
            
            # Test special tokens were added
            self.mock_tokenizer.add_special_tokens.assert_called()
    
    def test_dataset_getitem(self):
        """Test the __getitem__ method of the dataset"""
        # Patch audio file processing to avoid actual file loading
        with patch.object(AudioLLMDataset, '_process_audio', return_value=torch.randn(1, 80, 3000)):
            # Create dataset
            dataset = AudioLLMDataset(
                data_entries=self.data_entries,
                audio_dir=str(self.audio_dir),
                whisper_processor=self.mock_processor,
                llama_tokenizer=self.mock_tokenizer,
                max_audio_length=30,
                text_max_length=512
            )
            
            # Get first item
            item = dataset[0]
            
            # Check that keys exist
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            self.assertIn('audio_features', item)
            self.assertIn('labels', item)
            self.assertIn('text', item)
            self.assertIn('audio_path', item)
            
            # Check types
            self.assertIsInstance(item['input_ids'], torch.Tensor)
            self.assertIsInstance(item['attention_mask'], torch.Tensor)
            self.assertIsInstance(item['text'], str)
    
    def test_collate_fn(self):
        """Test the collate_fn function"""
        # Create sample batch items
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'audio_features': torch.randn(1, 80, 300),
                'labels': torch.tensor([4, 5, 6]),
            },
            {
                'input_ids': torch.tensor([7, 8, 9]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'audio_features': torch.randn(1, 80, 300),
                'labels': torch.tensor([10, 11, 12]),
            }
        ]
        
        # Apply collate function
        collated = collate_fn(batch)
        
        # Check output structure
        self.assertIn('input_ids', collated)
        self.assertIn('attention_mask', collated)
        self.assertIn('audio_features', collated)
        self.assertIn('labels', collated)
        self.assertIn('metadata', collated)
        
        # Check shapes
        self.assertEqual(collated['input_ids'].shape, (2, 3))
        self.assertEqual(collated['attention_mask'].shape, (2, 3))
        self.assertEqual(collated['audio_features'].shape, (2, 1, 80, 300))
        self.assertEqual(collated['labels'].shape, (2, 3))
        
    @patch('dataloaders.WhisperProcessor.from_pretrained')
    @patch('dataloaders.LlamaTokenizer.from_pretrained')
    def test_create_dataloaders(self, mock_tokenizer_from_pretrained, mock_processor_from_pretrained):
        """Test the create_dataloaders function"""
        # Setup mocks
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        mock_processor_from_pretrained.return_value = self.mock_processor
        
        # Patch dataset to avoid audio processing
        with patch.object(AudioLLMDataset, '_process_audio', return_value=torch.randn(1, 80, 3000)):
            # Call create_dataloaders
            train_loader, val_loader = create_dataloaders(
                data_path=str(self.data_path),
                audio_dir=str(self.audio_dir),
                batch_size=1,  # Small batch size for testing
                train_split=0.67,  # Split to ensure we have both train and val data
                num_workers=0  # Use 0 workers for testing
            )
            
            # Check that we have DataLoader instances
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            
            # Check basic properties of the loaders
            self.assertEqual(train_loader.batch_size, 1)
            self.assertEqual(val_loader.batch_size, 1)
            
            # Check that train and val loaders have different lengths
            # In a real test, we might need to handle edge cases better
            expected_train_size = max(1, int(len(self.data_entries) * 0.67))
            expected_val_size = max(1, len(self.data_entries) - expected_train_size)
            
            # Since our dataset is small, we check the lengths
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            
            self.assertEqual(train_size + val_size, len(self.data_entries))
    
    def test_get_sample_batch(self):
        """Test the get_sample_batch function"""
        # Create a mock dataloader
        mock_dataloader = MagicMock()
        mock_batch = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]]),
            'audio_features': torch.randn(2, 1, 80, 300),
            'labels': torch.tensor([[7, 8, 9], [10, 11, 12]]),
            'metadata': [{'key': 'value1'}, {'key': 'value2'}]
        }
        mock_dataloader.__iter__.return_value = iter([mock_batch])
        
        # Call get_sample_batch
        with patch('builtins.print'):  # Suppress print statements
            sample = get_sample_batch(mock_dataloader)
        
        # Check that we got the expected batch
        self.assertEqual(sample, mock_batch)


if __name__ == '__main__':
    unittest.main()
