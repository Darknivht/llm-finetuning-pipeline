#!/usr/bin/env python3
"""
Data preparation script for the LLM Fine-Tuning Pipeline.
Handles CSV, JSON, TSV formats with cleaning and splitting functionality.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend import setup_logging
from backend.data_utils import DataProcessor, generate_synthetic_data, load_huggingface_dataset
from backend.models import ModelLoader
from backend.logging_config import get_logger

logger = get_logger("data_prep")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for LLM fine-tuning")
    
    # Input/Output
    parser.add_argument("--input_file", type=str,
                       help="Path to input data file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed datasets")
    
    # Data format
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of text column in data")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of label column in data")
    parser.add_argument("--format", type=str, choices=["csv", "json", "tsv", "jsonl"],
                       help="Input data format (auto-detected if not specified)")
    
    # Data processing
    parser.add_argument("--clean_data", type=bool, default=True,
                       help="Clean text data")
    parser.add_argument("--lowercase", type=bool, default=False,
                       help="Convert text to lowercase")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum text length")
    parser.add_argument("--min_length", type=int, default=1,
                       help="Minimum text length")
    
    # Data splitting
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="Test set ratio")
    
    # Sampling
    parser.add_argument("--max_examples", type=int,
                       help="Maximum number of examples to process")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Tokenization
    parser.add_argument("--tokenize", type=bool, default=True,
                       help="Tokenize the data")
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model name for tokenizer")
    
    # Synthetic data generation
    parser.add_argument("--generate_synthetic", action="store_true",
                       help="Generate synthetic data for testing")
    parser.add_argument("--synthetic_samples", type=int, default=100,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--task_type", type=str, default="completion",
                       choices=["completion", "classification", "summarization"],
                       help="Task type for synthetic data")
    
    # HuggingFace dataset
    parser.add_argument("--hf_dataset", type=str,
                       help="HuggingFace dataset name to download and process")
    parser.add_argument("--hf_subset", type=str,
                       help="HuggingFace dataset subset")
    parser.add_argument("--hf_split", type=str,
                       help="HuggingFace dataset split")
    
    return parser.parse_args()


def validate_splits(train_split: float, val_split: float, test_split: float):
    """Validate that splits sum to 1.0."""
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def main():
    """Main data preparation function."""
    args = parse_args()
    
    # Set up logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_level="INFO", log_dir=str(output_dir / "logs"))
    logger.info("Starting data preparation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate splits
    validate_splits(args.train_split, args.val_split, args.test_split)
    
    try:
        # Generate synthetic data if requested
        if args.generate_synthetic:
            logger.info(f"Generating {args.synthetic_samples} synthetic samples")
            
            synthetic_file = output_dir / "synthetic_data.csv"
            generate_synthetic_data(
                output_file=synthetic_file,
                num_samples=args.synthetic_samples,
                task_type=args.task_type,
                format_type="csv"
            )
            
            logger.info(f"Synthetic data saved to {synthetic_file}")
            
            # Use synthetic data as input if no other input specified
            if not args.input_file and not args.hf_dataset:
                args.input_file = str(synthetic_file)
        
        # Load tokenizer if tokenization requested
        tokenizer = None
        if args.tokenize:
            logger.info(f"Loading tokenizer: {args.model_name}")
            tokenizer = ModelLoader.load_tokenizer(args.model_name)
        
        # Initialize data processor
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        
        # Process data based on source
        if args.hf_dataset:
            # Load from HuggingFace Hub
            logger.info(f"Loading HuggingFace dataset: {args.hf_dataset}")
            
            dataset_dict = load_huggingface_dataset(
                args.hf_dataset,
                subset=args.hf_subset,
                split=args.hf_split,
                text_column=args.text_column,
                label_column=args.label_column,
                cache_dir=str(output_dir / "hf_cache")
            )
            
            # Save the loaded dataset
            dataset_output_dir = output_dir / "processed_dataset"
            dataset_dict.save_to_disk(str(dataset_output_dir))
            logger.info(f"HuggingFace dataset saved to {dataset_output_dir}")
        
        elif args.input_file:
            # Process local file
            input_file = Path(args.input_file)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            logger.info(f"Processing data from {input_file}")
            
            # Prepare dataset
            dataset_dict = data_processor.prepare_dataset(
                file_path=input_file,
                output_dir=output_dir / "processed_dataset",
                text_column=args.text_column,
                label_column=args.label_column,
                train_ratio=args.train_split,
                val_ratio=args.val_split,
                test_ratio=args.test_split,
                clean_data=args.clean_data,
                max_examples=args.max_examples,
                random_seed=args.seed
            )
            
            logger.info("Data processing completed")
            
            # Print dataset statistics
            logger.info("Dataset Statistics:")
            for split_name, split_dataset in dataset_dict.items():
                logger.info(f"  {split_name}: {len(split_dataset)} samples")
                
                # Sample statistics
                if len(split_dataset) > 0:
                    sample = split_dataset[0]
                    logger.info(f"  {split_name} columns: {list(sample.keys())}")
        
        else:
            raise ValueError("Must specify either --input_file, --hf_dataset, or --generate_synthetic")
        
        # Create data summary
        summary_file = output_dir / "data_summary.json"
        
        summary = {
            "input_source": args.hf_dataset or args.input_file or "synthetic",
            "task_type": args.task_type if args.generate_synthetic else "unknown",
            "splits": {
                "train": args.train_split,
                "validation": args.val_split,
                "test": args.test_split
            },
            "processing": {
                "clean_data": args.clean_data,
                "tokenize": args.tokenize,
                "max_length": args.max_length,
                "min_length": args.min_length,
                "max_examples": args.max_examples
            },
            "columns": {
                "text_column": args.text_column,
                "label_column": args.label_column
            }
        }
        
        if 'dataset_dict' in locals():
            summary["dataset_sizes"] = {
                split_name: len(split_dataset)
                for split_name, split_dataset in dataset_dict.items()
            }
        
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data summary saved to {summary_file}")
        
        # Create sample data files for inspection
        if 'dataset_dict' in locals():
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            
            for split_name, split_dataset in dataset_dict.items():
                if len(split_dataset) > 0:
                    # Save first 10 samples as JSON for inspection
                    sample_data = []
                    for i in range(min(10, len(split_dataset))):
                        sample_data.append(split_dataset[i])
                    
                    sample_file = samples_dir / f"{split_name}_samples.json"
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, indent=2, default=str)
                    
                    logger.info(f"Sample data saved to {sample_file}")
        
        logger.info("Data preparation completed successfully!")
        
        # Print usage instructions
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETED")
        print("=" * 60)
        print(f"Processed dataset saved to: {output_dir / 'processed_dataset'}")
        print(f"Data summary: {summary_file}")
        print(f"Sample data: {output_dir / 'samples'}")
        print("\nTo use this data for training:")
        print(f"python train.py --data_path {output_dir / 'processed_dataset'} --output_dir ./checkpoints")
        print("\n" + "=" * 60)
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()