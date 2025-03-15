# test_integration.py
import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model and dataset
from models.audio_llm import AudioLLM
from dataset import AudioLLMDataset, collate_fn

class TestIntegration(unittest.TestCase):
    
    @patch('models.audio_llm.load_base_models')
    def setUp(self, mock_load_base_models):
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the base models
        self.mock_llama = MagicMock()
        self.mock_whisper = MagicMock()
        
        # Configure the mocks
        self.mock_llama.model.config.hidden_size = 4096
        self.mock_whisper.model.config.d_model = 1024
        
        # Set up the model's embed_tokens
        self.mock_llama.model.model.embed_tokens = MagicMock()
        self.mock_llama.model.model.embed_tokens.return_value = torch.zeros(2, 10, 4096)
        
        # Return the mocked models when load_base_models is called
        mock_load_base_models.return_value = (self.mock_llama, self.mock_whisper)
        
        # Patch the apply_lora_to_llama function
        with patch('models.audio_llm.apply_lora_to_llama') as self.mock_apply_lora:
            self.mock_apply_lora.return_value = {'layer1': MagicMock(), 'layer2': MagicMock()}
            
            # Create the model
            self.model = AudioLLM(llama_path="mock_llama", whisper_path="mock_whisper")
            
            # Add tokenizer attribute for testing
            self.model.tokenizer = MagicMock()
            self.model.audio_start_token = "<audio>"
            self.model.audio_end_token = "</audio>"
            
            self.model.tokenizer.convert_tokens_to_ids.side_effect = lambda x: 1 if x == "<audio>" else 2
        
        # Mock the dataset components
        self.mock_whisper_processor = MagicMock()
        self.mock_whisper_processor.return_value = MagicMock(input_features=torch.randn(1, 80, 3000))
        
        self.mock_llama_tokenizer = MagicMock()
        
        # Mock tokenizer return values
        mock_tokens = MagicMock()
        mock_tokens.input_ids = torch.randint(0, 100, (1, 10))
        mock_tokens.attention_mask = torch.ones(1, 10)
        self.mock_llama_tokenizer.return_value = mock_tokens
        
        # Create test data
        self.test_data = [
            {
                "text": "Describe this audio.",
                "audio_paths": "test1.wav",
                "response": "This is a piano playing."
            },
            {
                "text": "What's in this recording?",
                "audio_paths": "test2.wav",
                "response": "This is a person speaking."
            }
        ]
        
        # Create the dataset
        self.dataset = AudioLLMDataset(
            data_entries=self.test_data,
            audio_dir=self.temp_dir,
            whisper_processor=self.mock_whisper_processor,
            llama_tokenizer=self.mock_llama_tokenizer,
            text_max_length=512
        )
    
    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('torchaudio.load')
    def test_model_with_dataset(self, mock_load):
        """Test that the model can process data from the dataset"""
        # Mock audio loading
        mock_waveform = torch.randn(1, 16000 * 5)  # 5 seconds of audio
        mock_sample_rate = 16000
        mock_load.return_value = (mock_waveform, mock_sample_rate)
        
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test1.wav")
        with open(test_file, "w") as f:
            f.write("dummy audio content")
        
        # Get an item from the dataset
        with patch('os.path.join', return_value=test_file):
            dataset_item = self.dataset[0]
        
        # Prepare inputs for the model
        input_ids = dataset_item["input_ids"].unsqueeze(0)  # Add batch dimension
        attention_mask = dataset_item["attention_mask"].unsqueeze(0)
        audio_features = dataset_item["audio_features"].unsqueeze(0) if dataset_item["audio_features"] is not None else None
        labels = dataset_item["labels"].unsqueeze(0)
        
        # Mock the model's forward method
        expected_output = MagicMock()
        self.model.forward = MagicMock(return_value=expected_output)
        
        # Call the model
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            labels=labels
        )
        
        # Check that forward was called with the right arguments
        self.model.forward.assert_called_once()
        
        # Check that the output matches what we expect
        self.assertEqual(output, expected_output)
    
    def test_dataloader_with_model(self):
        """Test that a DataLoader works with the dataset and model"""
        from torch.utils.data import DataLoader
        
        # Patch the __getitem__ method to return a consistent structure
        def mock_getitem(index):
            return {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.ones(10),
                "audio_features": torch.randn(1, 80, 3000),
                "labels": torch.randint(0, 100, (10,)),
                "metadata": {"id": index}
            }
        
        with patch.object(self.dataset, '__getitem__', side_effect=mock_getitem):
            # Create a DataLoader
            dataloader = DataLoader(
                self.dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # Get a batch
            batch = next(iter(dataloader))
            
            # Check batch structure
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertIn("audio_features", batch)
            self.assertIn("labels", batch)
            self.assertIn("metadata", batch)
            
            # Check shapes
            self.assertEqual(batch["input_ids"].shape, (2, 10))
            self.assertEqual(batch["attention_mask"].shape, (2, 10))
            self.assertEqual(batch["audio_features"].shape, (2, 1, 80, 3000))
            self.assertEqual(batch["labels"].shape, (2, 10))
            
            # Mock the model's forward method
            expected_output = MagicMock()
            self.model.forward = MagicMock(return_value=expected_output)
            
            # Call the model with the batch
            output = self.model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                labels=batch["labels"]
            )
            
            # Check that forward was called with the right arguments
            self.model.forward.assert_called_once()
            
            # Check that the output matches what we expect
            self.assertEqual(output, expected_output)
    
    @patch('torch.nn.functional.cross_entropy')
    def test_training_step(self, mock_cross_entropy):
        """Test a training step with the model and dataset"""
        # Define a simple training step function
        def training_step(model, batch):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                audio_features=batch["audio_features"],
                labels=batch["labels"]
            )
            
            # Calculate loss (typically done inside the LLaMA model, but mocked here)
            logits = torch.randn(2, 10, 32000)  # Mock logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                batch["labels"].view(-1)
            )
            
            return loss
        
        # Prepare a batch
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "audio_features": torch.randn(2, 1, 80, 3000),
            "labels": torch.randint(0, 100, (2, 10)),
            "metadata": [{"id": 0}, {"id": 1}]
        }
        
        # Mock the model's forward method
        outputs = MagicMock()
        outputs.logits = torch.randn(2, 10, 32000)
        self.model.forward = MagicMock(return_value=outputs)
        
        # Mock the loss calculation
        mock_loss = torch.tensor(2.5)
        mock_cross_entropy.return_value = mock_loss
        
        # Call the training step
        loss = training_step(self.model, batch)
        
        # Check that the model was called with the batch
        self.model.forward.assert_called_once_with(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            audio_features=batch["audio_features"],
            labels=batch["labels"]
        )
        
        # Check that loss was calculated
        mock_cross_entropy.assert_called_once()
        
        # Check that the loss is what we expect
        self.assertEqual(loss, mock_loss)

if __name__ == "__main__":
    unittest.main()
