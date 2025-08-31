#!/usr/bin/env python3
"""
Training script for the LLM Fine-Tuning Pipeline.
Supports both full fine-tuning and PEFT/LoRA with comprehensive configuration.
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend import setup_logging, Config
from backend.models import ModelLoader
from backend.data_utils import DataProcessor
from backend.train_utils import create_training_arguments, setup_trainer, save_model_with_config
from backend.logging_config import get_logger

logger = get_logger("train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LLM with fine-tuning or PEFT/LoRA")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="distilgpt2",
                       help="Model name or path")
    parser.add_argument("--use_peft", type=bool, default=True,
                       help="Use PEFT/LoRA for efficient fine-tuning")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data file or directory")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of text column in data")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of label column in data")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Log every N steps")
    
    # PEFT/LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--fp16", type=bool, default=True,
                       help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Evaluation arguments
    parser.add_argument("--do_eval", type=bool, default=True,
                       help="Run evaluation during training")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    
    # Resuming
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Resume training from checkpoint")
    
    # Data preparation
    parser.add_argument("--clean_data", type=bool, default=True,
                       help="Clean text data before training")
    parser.add_argument("--max_examples", type=int,
                       help="Maximum number of training examples")
    
    return parser.parse_args()


def setup_device_and_seed(device: str, seed: int):
    """Set up device and random seed."""
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    
    return device


def load_and_prepare_data(args, tokenizer):
    """Load and prepare training data."""
    logger.info(f"Loading data from {args.data_path}")
    
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Check if data_path is a file or directory
    data_path = Path(args.data_path)
    
    if data_path.is_file():
        # Load from single file
        dataset_dict = data_processor.prepare_dataset(
            file_path=data_path,
            output_dir=Path(args.output_dir) / "processed_data",
            text_column=args.text_column,
            label_column=args.label_column if hasattr(args, 'label_column') else None,
            train_ratio=1.0 - args.val_split if args.do_eval else 1.0,
            val_ratio=args.val_split if args.do_eval else 0.0,
            test_ratio=0.0,
            clean_data=args.clean_data,
            max_examples=args.max_examples
        )
    elif data_path.is_dir():
        # Look for processed datasets
        if (data_path / "dataset_dict.json").exists():
            from datasets import load_from_disk
            dataset_dict = load_from_disk(str(data_path))
            logger.info(f"Loaded pre-processed dataset from {data_path}")
        else:
            # Look for data files in directory
            data_files = []
            for ext in ['.csv', '.json', '.jsonl', '.tsv']:
                data_files.extend(data_path.glob(f"*{ext}"))
            
            if not data_files:
                raise FileNotFoundError(f"No data files found in {data_path}")
            
            # Use the first data file found
            dataset_dict = data_processor.prepare_dataset(
                file_path=data_files[0],
                output_dir=Path(args.output_dir) / "processed_data",
                text_column=args.text_column,
                label_column=args.label_column if hasattr(args, 'label_column') else None,
                train_ratio=1.0 - args.val_split if args.do_eval else 1.0,
                val_ratio=args.val_split if args.do_eval else 0.0,
                test_ratio=0.0,
                clean_data=args.clean_data,
                max_examples=args.max_examples
            )
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    return dataset_dict


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level="INFO", log_dir=f"{args.output_dir}/logs")
    logger.info("Starting LLM fine-tuning training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration if provided
    config = None
    if args.config:
        config = Config.from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Set up device and seed
    device = setup_device_and_seed(args.device, args.seed)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = ModelLoader.load_tokenizer(args.model_name_or_path)
    
    # Load and prepare data
    dataset_dict = load_and_prepare_data(args, tokenizer)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation") if args.do_eval else None
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")
    
    # Load base model
    logger.info(f"Loading base model: {args.model_name_or_path}")
    model = ModelLoader.load_base_model(
        args.model_name_or_path,
        device=device
    )
    
    # Resize token embeddings if tokenizer was modified
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")
    
    # Set up PEFT if requested
    if args.use_peft:
        logger.info("Setting up PEFT/LoRA model")
        model = ModelLoader.setup_peft_model(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    
    # Create training arguments
    training_args = create_training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.do_eval else None,
        evaluation_strategy="steps" if args.do_eval else "no",
        fp16=args.fp16 and device != "cpu",
        load_best_model_at_end=args.do_eval
    )
    
    # Set up trainer
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming training from {args.resume_from_checkpoint}")
    
    try:
        # Start training
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model
        logger.info("Saving final model...")
        
        # Prepare config for saving
        training_config = {
            "model_name_or_path": args.model_name_or_path,
            "use_peft": args.use_peft,
            "max_length": args.max_length,
            "training_args": training_args.to_dict(),
            "final_loss": trainer.state.log_history[-1].get("train_loss", 0.0) if trainer.state.log_history else 0.0,
        }
        
        if args.use_peft:
            training_config.update({
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            })
        
        # Save model and tokenizer
        final_model_dir = output_dir / "final_model"
        save_model_with_config(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(final_model_dir),
            config_dict=training_config
        )
        
        logger.info(f"Training completed successfully! Model saved to {final_model_dir}")
        
        # Print training summary
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            logger.info("Training Summary:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        logger.info("Saving interrupted model state...")
        interrupted_dir = output_dir / "interrupted_model"
        save_model_with_config(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(interrupted_dir),
            config_dict={"status": "interrupted", "step": trainer.state.global_step}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()