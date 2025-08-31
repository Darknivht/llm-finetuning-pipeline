"""
Pytest configuration and fixtures for the LLM Fine-Tuning Pipeline tests.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add backend to Python path for testing
import sys
test_root = Path(__file__).parent.parent
sys.path.insert(0, str(test_root))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="llm_pipeline_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_training_data():
    """Generate sample training data for tests."""
    return [
        {"text": "The weather today is sunny and warm.", "label": "weather"},
        {"text": "Machine learning is a subset of artificial intelligence.", "label": "technology"},
        {"text": "Python is a popular programming language.", "label": "programming"},
        {"text": "The stock market closed higher today.", "label": "finance"},
        {"text": "Regular exercise is important for health.", "label": "health"},
    ]


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.config = Mock()
    mock.config.vocab_size = 50000
    mock.generate.return_value = [[1, 2, 3, 4, 5]]  # Mock token IDs
    return mock


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    mock = Mock()
    mock.pad_token_id = 0
    mock.eos_token_id = 1
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "Generated text"
    mock.__len__ = Mock(return_value=50000)
    
    # Mock tokenization call
    mock.return_value = {
        'input_ids': [[1, 2, 3, 4, 5]],
        'attention_mask': [[1, 1, 1, 1, 1]]
    }
    
    return mock


@pytest.fixture
def mock_openrouter_response():
    """Mock OpenRouter API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mocked response from OpenRouter API."
                }
            }
        ]
    }


@pytest.fixture
def temp_config_file(test_data_dir):
    """Create a temporary configuration file."""
    config_content = """
training:
  model_name_or_path: "distilgpt2"
  output_dir: "./test_output"
  num_train_epochs: 1
  per_device_train_batch_size: 2
  learning_rate: 2e-4
  use_peft: true
  lora_rank: 8

data:
  text_column: "text"
  max_examples: 10

eval:
  max_new_tokens: 20
  temperature: 0.7
"""
    
    config_file = test_data_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def temp_data_file(test_data_dir, sample_training_data):
    """Create a temporary CSV data file."""
    import csv
    
    data_file = test_data_dir / "test_data.csv"
    with open(data_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(sample_training_data)
    
    return data_file


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Mock model loading to avoid downloading models during tests."""
    with patch('backend.models.AutoModelForCausalLM.from_pretrained') as mock_model, \
         patch('backend.models.AutoTokenizer.from_pretrained') as mock_tokenizer:
        
        # Configure mock model
        mock_model_instance = Mock()
        mock_model_instance.config.vocab_size = 50000
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model.return_value = mock_model_instance
        
        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.__len__ = Mock(return_value=50000)
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.decode.return_value = "Generated text"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'model_instance': mock_model_instance,
            'tokenizer_instance': mock_tokenizer_instance
        }


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client for testing."""
    with patch('backend.openrouter.OpenRouterClient') as mock_client:
        mock_instance = Mock()
        mock_instance.is_available.return_value = True
        mock_instance.complete_text.return_value = "Mocked OpenRouter response"
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="session")
def playwright_browser_type():
    """Configure Playwright browser type for tests."""
    return "chromium"  # Use Chromium for consistency


@pytest.fixture(scope="session")  
def playwright_browser_options():
    """Configure Playwright browser options."""
    return {
        "headless": True,  # Run headless in CI
        "slow_mo": 100 if os.getenv("PLAYWRIGHT_DEBUG") else 0,  # Slow down for debugging
        "devtools": bool(os.getenv("PLAYWRIGHT_DEBUG")),
    }


@pytest.fixture(scope="session")
def playwright_context_options():
    """Configure Playwright context options."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add integration marker to tests in integration directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to E2E tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(pytest.mark.skip(reason="CUDA not available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))


@pytest.fixture
def set_env_vars():
    """Fixture to temporarily set environment variables for tests."""
    original_env = os.environ.copy()
    
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    yield _set_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs():
    """Fixture to capture log output during tests."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    yield log_capture
    
    # Cleanup
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)