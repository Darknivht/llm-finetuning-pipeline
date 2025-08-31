"""
Model loading and adapter utilities for the LLM Fine-Tuning Pipeline.
Supports both standard HuggingFace models and PEFT/LoRA adapters.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from peft import (
    PeftModel, 
    PeftConfig, 
    LoraConfig, 
    get_peft_model,
    TaskType
)

from .logging_config import get_logger

logger = get_logger("models")


class ModelLoader:
    """Utility class for loading models and tokenizers with PEFT support."""
    
    @staticmethod
    def load_tokenizer(
        model_name_or_path: str,
        padding_side: str = "right",
        add_special_tokens: bool = True
    ) -> AutoTokenizer:
        """
        Load tokenizer with proper configuration.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model ID
            padding_side: Side to pad tokens ("left" or "right")
            add_special_tokens: Whether to add special tokens
        
        Returns:
            Configured tokenizer
        """
        logger.info(f"Loading tokenizer from {model_name_or_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception as e:
            logger.warning(f"Failed to load AutoTokenizer, trying GPT2Tokenizer: {e}")
            tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Set padding side
        tokenizer.padding_side = padding_side
        
        logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
        return tokenizer
    
    @staticmethod
    def load_base_model(
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True
    ) -> AutoModelForCausalLM:
        """
        Load base model for training or inference.
        
        Args:
            model_name_or_path: Path to model or HuggingFace model ID
            device: Device to load model on
            torch_dtype: Data type for model weights
            low_cpu_mem_usage: Whether to use low CPU memory usage
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading base model from {model_name_or_path}")
        
        # Auto-detect dtype if not specified
        if torch_dtype is None:
            if device != "cpu" and torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load AutoModelForCausalLM, trying GPT2LMHeadModel: {e}")
            model = GPT2LMHeadModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage
            )
        
        # Move to device if specified
        if device != "auto":
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"
            model = model.to(device)
        
        logger.info(f"Model loaded with {model.num_parameters():,} parameters")
        return model
    
    @staticmethod
    def setup_peft_model(
        model: AutoModelForCausalLM,
        lora_config: Optional[LoraConfig] = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ) -> PeftModel:
        """
        Set up PEFT/LoRA model for efficient fine-tuning.
        
        Args:
            model: Base model to add adapters to
            lora_config: Pre-configured LoRA config
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
        
        Returns:
            PEFT model with LoRA adapters
        """
        logger.info("Setting up PEFT/LoRA model")
        
        if lora_config is None:
            # Default target modules for common architectures
            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                # Fallback for GPT2-style models
                if hasattr(model, 'transformer'):
                    target_modules = ["c_attn", "c_proj"]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
        
        peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        trainable_params = 0
        all_params = 0
        for _, param in peft_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_ratio = 100 * trainable_params / all_params
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
        logger.info(f"Total parameters: {all_params:,}")
        
        return peft_model
    
    @staticmethod
    def load_peft_model(
        base_model_path: str,
        adapter_path: str,
        device: str = "auto"
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Load a fine-tuned PEFT model and tokenizer.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to PEFT adapters
            device: Device to load model on
        
        Returns:
            Tuple of (PEFT model, tokenizer)
        """
        logger.info(f"Loading PEFT model: base={base_model_path}, adapter={adapter_path}")
        
        # Load tokenizer
        tokenizer = ModelLoader.load_tokenizer(base_model_path)
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(adapter_path)
        
        # Load base model
        base_model = ModelLoader.load_base_model(
            peft_config.base_model_name_or_path or base_model_path,
            device=device
        )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("PEFT model loaded successfully")
        return model, tokenizer
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: Union[str, Path],
        device: str = "auto"
    ) -> Tuple[Union[AutoModelForCausalLM, PeftModel], AutoTokenizer, Dict[str, Any]]:
        """
        Load model checkpoint (either standard or PEFT).
        
        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on
        
        Returns:
            Tuple of (model, tokenizer, config_dict)
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load run config if available
        config_file = checkpoint_path / "run_config.json"
        config_dict = {}
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        
        # Check if it's a PEFT checkpoint
        adapter_config_file = checkpoint_path / "adapter_config.json"
        
        if adapter_config_file.exists():
            # Load PEFT model
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            base_model_path = peft_config.base_model_name_or_path
            
            tokenizer = ModelLoader.load_tokenizer(base_model_path)
            base_model = ModelLoader.load_base_model(base_model_path, device=device)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            logger.info("Loaded PEFT checkpoint")
        else:
            # Load standard checkpoint
            tokenizer = ModelLoader.load_tokenizer(checkpoint_path)
            model = ModelLoader.load_base_model(checkpoint_path, device=device)
            
            logger.info("Loaded standard checkpoint")
        
        return model, tokenizer, config_dict


def get_model_info(model: Union[AutoModelForCausalLM, PeftModel]) -> Dict[str, Any]:
    """
    Get information about a loaded model.
    
    Args:
        model: Model to inspect
    
    Returns:
        Dictionary with model information
    """
    info = {}
    
    # Basic info
    info['model_type'] = model.__class__.__name__
    info['is_peft'] = isinstance(model, PeftModel)
    
    # Parameter counts
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    info['total_parameters'] = total_params
    info['trainable_parameters'] = trainable_params
    info['trainable_percentage'] = 100 * trainable_params / total_params if total_params > 0 else 0
    
    # Memory info
    if hasattr(model, 'get_memory_footprint'):
        try:
            info['memory_footprint_mb'] = model.get_memory_footprint() / 1024 / 1024
        except:
            info['memory_footprint_mb'] = None
    
    # Device info
    device = next(model.parameters()).device
    info['device'] = str(device)
    info['dtype'] = str(next(model.parameters()).dtype)
    
    return info