"""
Unit tests for the models module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from backend.models import ModelLoader


class TestModelLoader:
    """Test ModelLoader functionality."""
    
    def test_load_tokenizer_success(self, mock_model_loading):
        """Test successful tokenizer loading."""
        tokenizer = ModelLoader.load_tokenizer("test-model")
        
        assert tokenizer is not None
        mock_model_loading['tokenizer'].assert_called_once()
    
    def test_load_base_model_success(self, mock_model_loading):
        """Test successful base model loading."""
        model = ModelLoader.load_base_model("test-model", device="cpu")
        
        assert model is not None
        mock_model_loading['model'].assert_called_once()
    
    def test_setup_peft_model(self, mock_model_loading):
        """Test PEFT model setup."""
        with patch('backend.models.get_peft_model') as mock_get_peft:
            mock_peft_model = Mock()
            mock_get_peft.return_value = mock_peft_model
            
            base_model = Mock()
            peft_model = ModelLoader.setup_peft_model(
                base_model, 
                lora_rank=16, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
            
            assert peft_model == mock_peft_model
            mock_get_peft.assert_called_once()
    
    @patch('backend.models.Path.exists')
    @patch('backend.models.json.load')
    @patch('builtins.open')
    def test_load_checkpoint_success(self, mock_open, mock_json_load, mock_exists, mock_model_loading):
        """Test successful checkpoint loading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock config loading
        mock_config = {"model_type": "peft", "use_peft": True}
        mock_json_load.return_value = mock_config
        
        with patch('backend.models.PeftModel.from_pretrained') as mock_peft_load:
            mock_peft_model = Mock()
            mock_peft_load.return_value = mock_peft_model
            
            model, tokenizer, config = ModelLoader.load_checkpoint("test-checkpoint")
            
            assert model == mock_peft_model
            assert tokenizer is not None
            assert config == mock_config
    
    @patch('backend.models.Path.exists')
    def test_load_checkpoint_not_found(self, mock_exists):
        """Test checkpoint loading when checkpoint doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            ModelLoader.load_checkpoint("non-existent-checkpoint")
    
    def test_auto_device_selection_cuda(self):
        """Test automatic device selection with CUDA available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = ModelLoader._determine_device("auto")
            assert device == "cuda"
    
    def test_auto_device_selection_cpu(self):
        """Test automatic device selection with CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            device = ModelLoader._determine_device("auto")
            assert device == "cpu"
    
    def test_explicit_device_selection(self):
        """Test explicit device selection."""
        device = ModelLoader._determine_device("cpu")
        assert device == "cpu"
        
        device = ModelLoader._determine_device("cuda")
        assert device == "cuda"
    
    def test_model_to_device(self, mock_model_loading):
        """Test moving model to device."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        
        result = ModelLoader._move_model_to_device(mock_model, "cpu")
        
        mock_model.to.assert_called_once_with("cpu")
        assert result == mock_model
    
    @patch('backend.models.logging')
    def test_error_handling_in_load_tokenizer(self, mock_logging, mock_model_loading):
        """Test error handling in tokenizer loading."""
        mock_model_loading['tokenizer'].side_effect = Exception("Loading failed")
        
        with pytest.raises(Exception):
            ModelLoader.load_tokenizer("test-model")
    
    @patch('backend.models.logging')
    def test_error_handling_in_load_model(self, mock_logging, mock_model_loading):
        """Test error handling in model loading."""
        mock_model_loading['model'].side_effect = Exception("Loading failed")
        
        with pytest.raises(Exception):
            ModelLoader.load_base_model("test-model")
    
    def test_peft_config_creation(self):
        """Test PEFT configuration creation."""
        with patch('backend.models.LoraConfig') as mock_lora_config:
            mock_config = Mock()
            mock_lora_config.return_value = mock_config
            
            config = ModelLoader._create_peft_config(
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["attention"]
            )
            
            assert config == mock_config
            mock_lora_config.assert_called_once()
    
    def test_save_model_components(self, mock_model_loading, test_data_dir):
        """Test saving model components."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        output_dir = test_data_dir / "test_save"
        
        # Mock the save methods
        mock_model.save_pretrained = Mock()
        mock_tokenizer.save_pretrained = Mock()
        
        ModelLoader.save_model_components(mock_model, mock_tokenizer, str(output_dir))
        
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
    
    def test_validate_checkpoint_structure(self):
        """Test checkpoint structure validation."""
        with patch('backend.models.Path') as mock_path:
            # Mock valid checkpoint structure
            mock_checkpoint_path = Mock()
            mock_path.return_value = mock_checkpoint_path
            mock_checkpoint_path.exists.return_value = True
            mock_checkpoint_path.is_dir.return_value = True
            
            # Mock required files
            config_file = Mock()
            config_file.exists.return_value = True
            mock_checkpoint_path.__truediv__ = Mock(return_value=config_file)
            
            result = ModelLoader._validate_checkpoint_structure(mock_checkpoint_path)
            assert result is True
    
    def test_get_model_info(self, mock_model_loading):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.config.model_type = "gpt2"
        mock_model.config.vocab_size = 50000
        
        info = ModelLoader.get_model_info(mock_model)
        
        assert "model_type" in info
        assert "vocab_size" in info
        assert info["model_type"] == "gpt2"
        assert info["vocab_size"] == 50000