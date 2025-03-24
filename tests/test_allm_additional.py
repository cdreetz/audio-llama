import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src directory to path to import modules
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(src_path)

from models.allm import AudioLLM

class TestAudioLLMAdditional(unittest.TestCase):
    
    @patch('models.allm.load_base_models')
    def setUp(self, mock_load_base_models):
        # Similar setup as in test_allm.py
        self.mock_llama = MagicMock()
        self.mock_whisper = MagicMock()
        self.mock_llama.model.config.hidden_size = 4096
        self.mock_whisper.model.config.d_model = 1024
        self.mock_llama.model.model.embed_tokens = MagicMock()
        mock_load_base_models.return_value = (self.mock_llama, self.mock_whisper)
        
        with patch('models.allm.apply_lora_to_llama') as self.mock_apply_lora:
            self.mock_apply_lora.return_value = {'layer1': MagicMock()}
            self.model = AudioLLM(llama_path="mock_llama", whisper_path="mock_whisper")
            self.model.tokenizer = MagicMock()

    def test_process_audio_features(self):
        """Test the audio processing pipeline through conv layers"""
        batch_size = 2
        seq_len = 160  # Example sequence length
        
        # Input shape: [batch_size, channels, features, time]
        audio_features = torch.randn(batch_size, 1, 80, seq_len)  # 1 channel, 80 mel features, seq_len time steps
        
        processed = self.model._process_audio_features(audio_features)
        
        # Check final output shape
        # After 3 conv layers with stride 2, sequence length should be reduced by factor of 8
        expected_seq_len = seq_len // 8
        self.assertEqual(processed.shape, (batch_size, expected_seq_len, 128))
        
        # Check that output contains no negative values (due to ReLU)
        self.assertTrue(torch.all(processed >= 0))

    def test_extend_attention_mask_no_special_tokens(self):
        """Test attention mask extension without special tokens"""
        batch_size = 2
        text_seq_len = 10
        audio_seq_len = 5
        
        attention_mask = torch.ones(batch_size, text_seq_len)
        extended_mask = self.model._extend_attention_mask(
            attention_mask, 
            audio_seq_len,
            has_special_tokens=False
        )
        
        expected_shape = (batch_size, text_seq_len + audio_seq_len)
        self.assertEqual(extended_mask.shape, expected_shape)

    def test_token_embedding_error_handling(self):
        """Test error handling for invalid token IDs"""
        batch_size = 2
        text_seq_len = 10
        audio_seq_len = 5
        hidden_dim = 4096
        
        # Mock tokenizer to return invalid token IDs
        self.model.tokenizer.convert_tokens_to_ids.return_value = 50000  # Invalid ID
        
        # Mock embed_tokens vocabulary size
        self.model.llama.model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.randn(1000, hidden_dim)  # Vocab size of 1000
        )
        
        text_embeddings = torch.randn(batch_size, text_seq_len, hidden_dim)
        audio_features = torch.randn(batch_size, audio_seq_len, 1024)
        input_ids = torch.randint(0, 100, (batch_size, text_seq_len))
        
        # Should raise ValueError due to invalid token IDs
        with self.assertRaises(ValueError):
            self.model._combine_text_and_audio_embeddings(
                text_embeddings, 
                audio_features,
                input_ids
            )

    def test_lora_hooks(self):
        """Test LoRA hook registration and cleanup"""
        # Count initial hooks
        initial_hooks = len(self.model.hooks)
        
        # Remove hooks
        for hook in self.model.hooks:
            hook.remove()
        
        # Verify hooks are removed
        remaining_hooks = sum(1 for _ in self.model.llama.model.modules() if len(list(_._forward_hooks)) > 0)
        self.assertEqual(remaining_hooks, 0)
        
        # Re-register hooks
        self.model.hooks = []
        for name, module in self.model.llama.model.named_modules():
            if name in self.model.lora_layers:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self.model.lora_layers[n](out)
                )
                self.model.hooks.append(hook)
        
        # Verify hooks are re-registered
        self.assertEqual(len(self.model.hooks), initial_hooks)

    def test_device_handling(self):
        """Test handling of device mismatches"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        batch_size = 2
        text_seq_len = 10
        hidden_dim = 4096
        
        # Put model on CPU
        self.model = self.model.cpu()
        
        # Create CUDA tensors
        input_ids = torch.randint(0, 100, (batch_size, text_seq_len)).cuda()
        attention_mask = torch.ones(batch_size, text_seq_len).cuda()
        
        # Mock llama model to track device movement
        self.model.llama.model = MagicMock()
        self.model.llama.model.to = MagicMock()
        
        # Run forward pass
        self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        # Verify model was moved to CUDA
        self.model.llama.model.to.assert_called_with(input_ids.device)

if __name__ == '__main__':
    unittest.main()
