"""
LLM Fine-Tuning Pipeline Backend Package
=======================================

This package provides modular components for fine-tuning, evaluating,
and deploying language models with support for:
- Full fine-tuning and LoRA/PEFT
- Multiple data formats (CSV, JSON, TSV)
- OpenRouter fallback for inference
- Production-ready logging and configuration
"""

__version__ = "1.0.0"
__author__ = "LLM Fine-Tuning Pipeline"

from .config import Config
from .logging_config import setup_logging

__all__ = ["Config", "setup_logging"]