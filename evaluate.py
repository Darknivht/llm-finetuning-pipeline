#!/usr/bin/env python3
"""
Evaluation script for the LLM Fine-Tuning Pipeline.
Supports comprehensive evaluation with multiple metrics and OpenRouter comparison.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend import setup_logging, Config
from backend.models import ModelLoader
from backend.data_utils import DataProcessor, load_huggingface_dataset
from backend.eval_utils import evaluate_model, TextGenerationEvaluator, MetricsComputer
from backend.openrouter import create_fallback_client
from backend.logging_config import get_logger

logger = get_logger("evaluate")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--base_model", type=str,
                       help="Base model path (for PEFT models)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to evaluation dataset or HuggingFace dataset name")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of text column in data")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of label column in data")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on")
    parser.add_argument("--sample_size", type=int,
                       help="Number of samples to evaluate (None for all)")
    
    # Evaluation arguments
    parser.add_argument("--task_type", type=str, default="generation",
                       choices=["generation", "classification", "summarization"],
                       help="Type of task for evaluation")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--compute_perplexity", type=bool, default=True,
                       help="Compute model perplexity")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--do_sample", type=bool, default=True,
                       help="Whether to use sampling")
    
    # Comparison arguments
    parser.add_argument("--compare_with_openrouter", type=bool, default=False,
                       help="Compare results with OpenRouter model")
    parser.add_argument("--openrouter_model", type=str, default="openrouter/auto",
                       help="OpenRouter model to compare with")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    
    return parser.parse_args()


def load_evaluation_dataset(
    dataset_path: str,
    text_column: str,
    label_column: Optional[str],
    split: str = "test",
    sample_size: Optional[int] = None
):
    """Load evaluation dataset from file or HuggingFace Hub."""
    logger.info(f"Loading evaluation dataset: {dataset_path}")
    
    dataset_path = Path(dataset_path)
    
    if dataset_path.exists():
        # Load from local file or directory
        if dataset_path.is_file():
            # Single file
            data_processor = DataProcessor()
            raw_data = data_processor.load_data_file(
                dataset_path, text_column, label_column
            )
            
            # Convert to dataset
            from datasets import Dataset
            eval_dataset = Dataset.from_list(raw_data)
        
        elif dataset_path.is_dir():
            # Directory with processed datasets
            from datasets import load_from_disk
            dataset_dict = load_from_disk(str(dataset_path))
            
            if split in dataset_dict:
                eval_dataset = dataset_dict[split]
            else:
                available_splits = list(dataset_dict.keys())
                logger.warning(f"Split '{split}' not found. Available splits: {available_splits}")
                eval_dataset = dataset_dict[available_splits[0]]
                logger.info(f"Using split '{available_splits[0]}'")
        
        else:
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    else:
        # Try loading from HuggingFace Hub
        try:
            dataset_dict = load_huggingface_dataset(
                str(dataset_path),
                split=split,
                text_column=text_column,
                label_column=label_column
            )
            eval_dataset = dataset_dict[split]
            logger.info(f"Loaded dataset from HuggingFace Hub: {dataset_path}")
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    # Sample dataset if requested
    if sample_size and len(eval_dataset) > sample_size:
        import random
        indices = random.sample(range(len(eval_dataset)), sample_size)
        eval_dataset = eval_dataset.select(indices)
        logger.info(f"Sampled {sample_size} examples from dataset")
    
    logger.info(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
    return eval_dataset


def evaluate_with_openrouter(
    eval_dataset,
    openrouter_model: str,
    max_tokens: int = 50,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Evaluate using OpenRouter for comparison."""
    logger.info(f"Evaluating with OpenRouter model: {openrouter_model}")
    
    try:
        # Create OpenRouter client
        from backend.openrouter import OpenRouterClient, OpenRouterConfig
        
        openrouter_config = OpenRouterConfig(
            model=openrouter_model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        with OpenRouterClient(openrouter_config) as client:
            # Check if API is available
            if not client.is_available():
                logger.warning("OpenRouter API not available")
                return {"error": "OpenRouter API not available"}
            
            # Get inputs from dataset
            inputs = []
            references = []
            
            for item in eval_dataset:
                if isinstance(item, dict):
                    text = item.get("text", item.get("input", ""))
                    label = item.get("label", item.get("target", ""))
                else:
                    text = str(item)
                    label = ""
                
                inputs.append(text)
                references.append(label)
            
            # Generate predictions
            logger.info(f"Generating {len(inputs)} predictions with OpenRouter...")
            predictions = client.batch_complete(
                inputs,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Filter out None predictions
            valid_predictions = []
            valid_references = []
            
            for pred, ref in zip(predictions, references):
                if pred is not None:
                    valid_predictions.append(pred)
                    valid_references.append(ref)
            
            if not valid_predictions:
                logger.error("No valid predictions from OpenRouter")
                return {"error": "No valid predictions"}
            
            # Compute metrics
            metrics_computer = MetricsComputer()
            task_type = "generation"  # Default for OpenRouter comparison
            
            metrics = metrics_computer.compute_all_metrics(
                valid_predictions, valid_references, task_type
            )
            
            return {
                "model": openrouter_model,
                "num_predictions": len(valid_predictions),
                "success_rate": len(valid_predictions) / len(predictions) * 100,
                "metrics": metrics,
                "sample_predictions": [
                    {
                        "input": inputs[i],
                        "prediction": valid_predictions[i] if i < len(valid_predictions) else None,
                        "reference": valid_references[i] if i < len(valid_references) else ""
                    }
                    for i in range(min(5, len(inputs)))  # First 5 samples
                ]
            }
    
    except Exception as e:
        logger.error(f"OpenRouter evaluation failed: {e}")
        return {"error": str(e)}


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_level="INFO", log_dir=str(output_dir / "logs"))
    logger.info("Starting LLM model evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model, tokenizer, config_dict = ModelLoader.load_checkpoint(
        checkpoint_path, device=device
    )
    
    logger.info("Model loaded successfully")
    if config_dict:
        logger.info(f"Model configuration: {json.dumps(config_dict, indent=2, default=str)}")
    
    # Load evaluation dataset
    eval_dataset = load_evaluation_dataset(
        args.dataset,
        args.text_column,
        args.label_column,
        args.split,
        args.sample_size
    )
    
    # Prepare generation config
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "batch_size": args.batch_size
    }
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    
    try:
        results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            task_type=args.task_type,
            output_dir=str(output_dir),
            generation_config=generation_config,
            compute_perplexity=args.compute_perplexity,
            sample_size=args.sample_size
        )
        
        logger.info("Model evaluation completed")
        
        # Print key metrics
        if "metrics" in results:
            logger.info("Evaluation Metrics:")
            for metric, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        if "perplexity" in results:
            logger.info(f"  perplexity: {results['perplexity']:.4f}")
        
        # OpenRouter comparison if requested
        if args.compare_with_openrouter:
            logger.info("Running OpenRouter comparison...")
            openrouter_results = evaluate_with_openrouter(
                eval_dataset,
                args.openrouter_model,
                args.max_new_tokens,
                args.temperature
            )
            
            results["openrouter_comparison"] = openrouter_results
            
            if "error" not in openrouter_results:
                logger.info("OpenRouter Comparison:")
                for metric, value in openrouter_results.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  openrouter_{metric}: {value:.4f}")
        
        # Save final results
        results_file = output_dir / "final_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        # Create summary report
        summary_file = output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LLM Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Task Type: {args.task_type}\n")
            f.write(f"Samples Evaluated: {results['num_samples']}\n\n")
            
            if "perplexity" in results:
                f.write(f"Perplexity: {results['perplexity']:.4f}\n")
            
            if "metrics" in results:
                f.write("\nMetrics:\n")
                for metric, value in results["metrics"].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
            
            if "openrouter_comparison" in results and "error" not in results["openrouter_comparison"]:
                f.write(f"\nOpenRouter Comparison ({args.openrouter_model}):\n")
                for metric, value in results["openrouter_comparison"].get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
        
        logger.info(f"Evaluation summary saved to {summary_file}")
        logger.info("Evaluation completed successfully!")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()