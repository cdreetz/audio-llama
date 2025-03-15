# test_audio_llm.py
import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from models.allm import AudioLLM

class TestAudioLLM(unittest.TestCase):
    
    @patch('models.allm.load_base_models')
    def setUp(self, mock_load_base_models):
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
        with patch('models.allm.apply_lora_to_llama') as self.mock_apply_lora:
            self.mock_apply_lora.return_value = {'layer1': MagicMock(), 'layer2': MagicMock()}
            
            # Create the model
            self.model = AudioLLM(llama_path="mock_llama", whisper_path="mock_whisper")
            
            # Add tokenizer attribute for testing
            self.model.tokenizer = MagicMock()
            self.model.audio_start_token = "<audio>"
            self.model.audio_end_token = "</audio>"
            
            self.model.tokenizer.convert_tokens_to_ids.side_effect = lambda x: 1 if x == "<audio>" else 2

    def test_init(self):
        """Test that the model initializes correctly"""
        # Check that base models were loaded
        self.assertIsNotNone(self.model.llama)
        self.assertIsNotNone(self.model.whisper_encoder)
        
        # Check that projector was created
        self.assertIsNotNone(self.model.projector)
        
        # Check that LoRA was applied
        self.mock_apply_lora.assert_called_once()
        self.assertEqual(len(self.model.lora_layers), 2)
    
    def test_get_trainable_params(self):
        """Test that only projector and LoRA params are trainable"""
        # Mock parameters
        self.model.projector.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
        self.model.lora_layers['layer1'].parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
        self.model.lora_layers['layer2'].parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(10))])
        
        params = self.model.get_trainable_params()
        # We should have 3 parameters (1 from projector, 2 from LoRA layers)
        self.assertEqual(len(params), 3)
    
    def test_extend_attention_mask(self):
        """Test attention mask extension for audio"""
        batch_size = 2
        text_seq_len = 10
        audio_seq_len = 5
        
        # Create a mock attention mask
        attention_mask = torch.ones(batch_size, text_seq_len)
        
        # Call the method
        extended_mask = self.model._extend_attention_mask(attention_mask, audio_seq_len)
        
        # Check the shape and values
        expected_shape = (batch_size, text_seq_len + audio_seq_len + 2)  # +2 for audio start/end tokens
        self.assertEqual(extended_mask.shape, expected_shape)
        self.assertTrue(torch.all(extended_mask == 1))  # All values should be 1
    
    def test_combine_embeddings_with_audio(self):
        """Test combining text and audio embeddings"""
        batch_size = 2
        text_seq_len = 10
        audio_seq_len = 5
        hidden_dim = 4096
        
        # Create mock inputs
        text_embeddings = torch.randn(batch_size, text_seq_len, hidden_dim)
        audio_features = torch.randn(batch_size, audio_seq_len, 1024)  # Whisper dim
        input_ids = torch.randint(0, 100, (batch_size, text_seq_len))

        # create a projected audio tensor 
        projected_audio = torch.randn(batch_size, audio_seq_len, hidden_dim)

        # save original method to restore later
        original_method = self.model.projector.forward

        original_embed_tokens = self.model.llama.model.model.embed_tokens

        try:
            mock_forward = MagicMock(return_value=projected_audio)
            self.model.projector.forward = mock_forward

            def mock_embed_tokens(token_ids):
                if token_ids.shape[1] == 1:
                    return torch.randn(batch_size, 1, hidden_dim)

                else:
                    return text_embeddings

            self.model.llama.model.model.embed_tokens = mock_embed_tokens

            combined = self.model._combine_text_and_audio_embeddings(text_embeddings, audio_features, input_ids)

            self.model.projector.forward.assert_called_once_with(audio_features)

            # check the shape
            expected_shape = (batch_size, 1 + audio_seq_len + 1 + text_seq_len, hidden_dim)
            self.assertEqual(combined.shape, expected_shape)

        finally:
            self.model.projector.forward = original_method
            self.model.llama.model.model.embed_tokens = original_embed_tokens
        

    def test_forward_with_audio(self):
        """Test forward pass with audio input"""
        batch_size = 2
        text_seq_len = 10
        audio_seq_len = 5
        hidden_dim = 4096
        
        # Create mock inputs
        input_ids = torch.randint(0, 100, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        audio_features = torch.randn(batch_size, audio_seq_len, 1024)  # Whisper dim
        labels = torch.randint(0, 100, (batch_size, text_seq_len))
        
        # Mock methods
        self.model._combine_text_and_audio_embeddings = MagicMock()
        combined_embeddings = torch.randn(batch_size, text_seq_len + audio_seq_len + 2, hidden_dim)
        self.model._combine_text_and_audio_embeddings.return_value = combined_embeddings
        
        self.model._extend_attention_mask = MagicMock()
        extended_mask = torch.ones(batch_size, text_seq_len + audio_seq_len + 2)
        self.model._extend_attention_mask.return_value = extended_mask
        
        self.model.llama.model = MagicMock()
        expected_output = MagicMock()
        self.model.llama.model.return_value = expected_output
        
        # Call forward
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            labels=labels
        )
        
        # Check that internal methods were called correctly
        self.model.llama.model.model.embed_tokens.assert_called_once_with(input_ids)
        self.model._combine_text_and_audio_embeddings.assert_called_once()
        self.model._extend_attention_mask.assert_called_once()
        
        # Check that LLaMA model was called with combined embeddings and extended mask
        self.model.llama.model.assert_called_once()
        
        # Check output
        self.assertEqual(output, expected_output)
    
    def test_forward_text_only(self):
        """Test forward pass with text only (no audio)"""
        batch_size = 2
        text_seq_len = 10
        hidden_dim = 4096
        
        # Create mock inputs
        input_ids = torch.randint(0, 100, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 100, (batch_size, text_seq_len))
        
        # Mock embed_tokens to return a specific tensor
        text_embeddings = torch.randn(batch_size, text_seq_len, hidden_dim)
        self.model.llama.model.model.embed_tokens.return_value = text_embeddings
        
        self.model.llama.model = MagicMock()
        expected_output = MagicMock()
        self.model.llama.model.return_value = expected_output
        
        # Call forward without audio_features
        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check that LLaMA model was called with text embeddings
        self.model.llama.model.assert_called_once()
        
        # Check output
        self.assertEqual(output, expected_output)

if __name__ == "__main__":
    unittest.main()
