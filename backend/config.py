"""
Configuration management for the LLM Fine-Tuning Pipeline.
Supports loading from YAML/JSON files with environment variable overrides.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_name_or_path: str = "distilgpt2"
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 512
    use_peft: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    device: str = "auto"
    fp16: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_path: str = "./data"
    text_column: str = "text"
    label_column: str = "label"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_examples: Optional[int] = None
    cache_dir: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 8
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 1
    do_sample: bool = True
    compute_metrics: bool = True


@dataclass
class APIConfig:
    """API configuration for OpenAI/OpenRouter."""
    openai_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/auto"
    use_openrouter_fallback: bool = True
    max_retries: int = 3
    timeout: int = 30


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                raw_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                raw_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls._from_dict(raw_config)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary."""
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        eval_config = EvalConfig(**config_dict.get('eval', {}))
        api_config = APIConfig(**config_dict.get('api', {}))
        
        return cls(
            training=training_config,
            data=data_config,
            eval=eval_config,
            api=api_config
        )
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Training environment overrides
        self.training.model_name_or_path = os.getenv('MODEL_NAME', self.training.model_name_or_path)
        self.training.output_dir = os.getenv('OUTPUT_DIR', self.training.output_dir)
        self.training.device = os.getenv('DEVICE', self.training.device)
        
        if os.getenv('BATCH_SIZE'):
            batch_size = int(os.getenv('BATCH_SIZE'))
            self.training.per_device_train_batch_size = batch_size
            self.training.per_device_eval_batch_size = batch_size
        
        if os.getenv('LR'):
            self.training.learning_rate = float(os.getenv('LR'))
        
        if os.getenv('EPOCHS'):
            self.training.num_train_epochs = int(os.getenv('EPOCHS'))
        
        if os.getenv('MAX_LENGTH'):
            self.training.max_seq_length = int(os.getenv('MAX_LENGTH'))
        
        if os.getenv('USE_PEFT'):
            self.training.use_peft = os.getenv('USE_PEFT').lower() in ['true', '1', 'yes']
        
        if os.getenv('LORA_RANK'):
            self.training.lora_rank = int(os.getenv('LORA_RANK'))
        
        # Data environment overrides
        self.data.data_path = os.getenv('DATA_PATH', self.data.data_path)
        
        # API environment overrides
        self.api.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.api.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', self.api.openrouter_base_url)
        self.api.openrouter_model = os.getenv('OPENROUTER_MODEL', self.api.openrouter_model)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'eval': self.eval.__dict__,
            'api': self.api.__dict__
        }
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration with the following priority:
    1. From config file (if provided)
    2. Environment variables override
    3. Default values
    """
    if config_path:
        config = Config.from_file(config_path)
    else:
        config = Config()
    
    config.update_from_env()
    return config