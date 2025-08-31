"""
Training utilities for the LLM Fine-Tuning Pipeline.
Includes callbacks, schedulers, and training helpers.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
import time

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import EvalPrediction
from peft import PeftModel

from .logging_config import get_logger

logger = get_logger("train_utils")


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: float
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch_time: Optional[float] = None
    total_time: Optional[float] = None


class MetricsLogger(TrainerCallback):
    """Custom callback for logging training metrics."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize metrics logger."""
        self.log_dir = Path(log_dir) if log_dir else None
        self.metrics_history = []
        self.start_time = None
        self.epoch_start_time = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        logger.info("Training started")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
    
    def on_log(self, args, state, control, **kwargs):
        """Called when logging occurs."""
        if state.log_history:
            log_entry = state.log_history[-1]
            
            # Calculate epoch time
            epoch_time = None
            if self.epoch_start_time:
                epoch_time = time.time() - self.epoch_start_time
            
            # Calculate total time
            total_time = None
            if self.start_time:
                total_time = time.time() - self.start_time
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=log_entry.get('epoch', 0),
                step=log_entry.get('step', 0),
                train_loss=log_entry.get('train_loss', 0.0),
                eval_loss=log_entry.get('eval_loss'),
                eval_perplexity=log_entry.get('eval_perplexity'),
                learning_rate=log_entry.get('learning_rate'),
                epoch_time=epoch_time,
                total_time=total_time
            )
            
            self.metrics_history.append(metrics)
            
            # Log to file if specified
            if self.log_dir:
                metrics_file = self.log_dir / "training_metrics.jsonl"
                with open(metrics_file, 'a', encoding='utf-8') as f:
                    json.dump(asdict(metrics), f, default=str)
                    f.write('\n')
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final metrics summary
        if self.log_dir:
            summary_file = self.log_dir / "training_summary.json"
            summary = {
                "total_epochs": len(self.metrics_history),
                "total_steps": self.metrics_history[-1].step if self.metrics_history else 0,
                "total_time_seconds": total_time,
                "final_train_loss": self.metrics_history[-1].train_loss if self.metrics_history else None,
                "final_eval_loss": self.metrics_history[-1].eval_loss if self.metrics_history else None,
                "best_eval_loss": min(
                    [m.eval_loss for m in self.metrics_history if m.eval_loss is not None],
                    default=None
                )
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)


class CheckpointCallback(TrainerCallback):
    """Custom callback for advanced checkpointing."""
    
    def __init__(
        self,
        save_best: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        save_config: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_best: Whether to save the best model
            metric_for_best: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            save_config: Whether to save run configuration
        """
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.save_config = save_config
        self.best_metric = None
        self.best_step = None
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        if not self.save_best:
            return
        
        # Get current metric value
        current_metric = None
        if state.log_history:
            for log_entry in reversed(state.log_history):
                if self.metric_for_best in log_entry:
                    current_metric = log_entry[self.metric_for_best]
                    break
        
        if current_metric is None:
            return
        
        # Check if this is the best metric so far
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif self.greater_is_better:
            is_best = current_metric > self.best_metric
        else:
            is_best = current_metric < self.best_metric
        
        if is_best:
            self.best_metric = current_metric
            self.best_step = state.global_step
            
            # Save best model
            best_model_path = Path(args.output_dir) / "best_model"
            best_model_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer')
            
            if model:
                if isinstance(model, PeftModel):
                    model.save_pretrained(best_model_path)
                else:
                    model.save_pretrained(best_model_path)
            
            if tokenizer:
                tokenizer.save_pretrained(best_model_path)
            
            logger.info(f"Saved best model at step {self.best_step} with {self.metric_for_best}={self.best_metric:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.save_config:
            # Save training configuration
            config_data = {
                "model_name": getattr(args, 'model_name_or_path', 'unknown'),
                "training_args": args.to_dict(),
                "best_metric": self.best_metric,
                "best_step": self.best_step,
                "final_step": state.global_step,
                "num_train_epochs": args.num_train_epochs,
                "total_train_batch_size": args.train_batch_size * args.gradient_accumulation_steps * args.world_size,
            }
            
            config_file = Path(args.output_dir) / "run_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved run configuration to {config_file}")


def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for language modeling.
    
    Args:
        eval_preds: Evaluation predictions from trainer
    
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_preds
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Calculate perplexity
    # Filter out special tokens (typically -100 in labels)
    valid_labels = labels != -100
    if valid_labels.sum() == 0:
        perplexity = float('inf')
    else:
        # Get probabilities for valid positions
        valid_predictions = predictions[valid_labels]
        valid_labels_filtered = labels[valid_labels]
        
        # Calculate cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            torch.from_numpy(valid_predictions).float(),
            torch.from_numpy(valid_labels_filtered).long()
        )
        
        perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity
    }


def create_training_arguments(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    evaluation_strategy: str = "steps",
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    fp16: bool = True,
    dataloader_pin_memory: bool = False,
    remove_unused_columns: bool = False,
    **kwargs
) -> TrainingArguments:
    """
    Create TrainingArguments with sensible defaults.
    
    Args:
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Logging frequency
        save_steps: Save frequency
        eval_steps: Evaluation frequency
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model at end
        evaluation_strategy: Evaluation strategy
        metric_for_best_model: Metric for best model selection
        greater_is_better: Whether higher metric values are better
        fp16: Whether to use mixed precision training
        dataloader_pin_memory: Whether to pin memory in dataloader
        remove_unused_columns: Whether to remove unused columns
        **kwargs: Additional arguments
    
    Returns:
        Configured TrainingArguments
    """
    # Auto-detect device capabilities
    device_count = torch.cuda.device_count()
    if device_count == 0:
        fp16 = False  # Disable fp16 on CPU
        dataloader_pin_memory = False
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        evaluation_strategy=evaluation_strategy,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=fp16,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=remove_unused_columns,
        report_to=[],  # Disable wandb/tensorboard by default
        **kwargs
    )
    
    return training_args


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    compute_metrics_fn: Optional[Callable] = None,
    callbacks: Optional[List[TrainerCallback]] = None
) -> Trainer:
    """
    Set up Trainer with custom callbacks and configuration.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        compute_metrics_fn: Function to compute metrics
        callbacks: Additional callbacks
    
    Returns:
        Configured Trainer
    """
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks
    default_callbacks = [
        MetricsLogger(log_dir=training_args.output_dir),
        CheckpointCallback(
            save_best=True,
            metric_for_best=training_args.metric_for_best_model,
            greater_is_better=training_args.greater_is_better
        )
    ]
    
    # Add early stopping if evaluation is enabled
    if training_args.evaluation_strategy != "no":
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
        default_callbacks.append(early_stopping)
    
    all_callbacks = default_callbacks + callbacks
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn or compute_metrics,
        callbacks=all_callbacks
    )
    
    return trainer


def save_model_with_config(
    model,
    tokenizer,
    output_dir: str,
    config_dict: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model, tokenizer, and configuration.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        config_dict: Additional configuration to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if isinstance(model, PeftModel):
        model.save_pretrained(output_dir)
        # Also save base model info
        base_model_info = {
            "base_model_name_or_path": model.base_model.name_or_path if hasattr(model.base_model, 'name_or_path') else 'unknown',
            "peft_type": "LoRA",
            "task_type": "CAUSAL_LM"
        }
        
        with open(output_dir / "base_model_info.json", 'w') as f:
            json.dump(base_model_info, f, indent=2)
    else:
        model.save_pretrained(output_dir)
    
    # Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(output_dir)
    
    # Save additional configuration
    if config_dict:
        config_file = output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Model and configuration saved to {output_dir}")


def resume_training(
    checkpoint_dir: str,
    new_output_dir: Optional[str] = None,
    **training_kwargs
) -> Dict[str, Any]:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        new_output_dir: New output directory (optional)
        **training_kwargs: Additional training arguments
    
    Returns:
        Dictionary with resumed training information
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Find the latest checkpoint
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
    
    # Sort by step number
    def get_step_number(checkpoint_path):
        try:
            return int(checkpoint_path.name.split('-')[1])
        except (IndexError, ValueError):
            return 0
    
    latest_checkpoint = max(checkpoints, key=get_step_number)
    
    logger.info(f"Resuming training from {latest_checkpoint}")
    
    # Load run config if available
    config_file = checkpoint_path / "run_config.json"
    run_config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            run_config = json.load(f)
    
    resume_info = {
        "checkpoint_path": str(latest_checkpoint),
        "resume_from_checkpoint": True,
        "original_output_dir": str(checkpoint_path),
        "new_output_dir": new_output_dir or str(checkpoint_path),
        "run_config": run_config
    }
    
    return resume_info


def get_model_size(model) -> Dict[str, Any]:
    """
    Get model size information.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with size information
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    # Estimate memory usage (rough approximation)
    param_memory_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
    
    size_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
        "estimated_memory_mb": param_memory_mb,
        "parameter_size_category": "small" if total_params < 100e6 else "medium" if total_params < 1e9 else "large"
    }
    
    return size_info