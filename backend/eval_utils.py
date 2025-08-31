"""
Evaluation utilities for the LLM Fine-Tuning Pipeline.
Includes metrics computation and model evaluation functions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import re
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from .logging_config import get_logger

logger = get_logger("eval_utils")


class TextGenerationEvaluator:
    """Evaluator for text generation tasks."""
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str = "auto"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Language model for generation
            tokenizer: Tokenizer
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move model to device if not already there
        if hasattr(self.model, 'device') and str(self.model.device) != self.device:
            self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_text(
        self,
        inputs: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 1,
        do_sample: bool = True,
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate text from input prompts.
        
        Args:
            inputs: List of input prompts
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            batch_size: Batch size for generation
        
        Returns:
            List of generated texts
        """
        generations = []
        
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample and temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Process inputs in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            
            # Tokenize batch
            tokenized = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokenized,
                    generation_config=generation_config
                )
            
            # Decode generated sequences
            for j, output in enumerate(outputs):
                # Remove input tokens from output
                input_length = tokenized.input_ids[j].shape[0]
                generated_tokens = output[input_length:]
                
                # Decode
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                generations.append(generated_text.strip())
        
        return generations
    
    def evaluate_perplexity(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Calculate perplexity on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with perplexity metrics
        """
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        # Process dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            
            if isinstance(batch, dict):
                texts = batch.get("text", batch.get("input_text", []))
            else:
                texts = [item.get("text", item.get("input_text", "")) for item in batch]
            
            if not texts:
                continue
            
            # Tokenize batch
            tokenized = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.model(**tokenized, labels=tokenized.input_ids)
                loss = outputs.loss
                
                # Count tokens (excluding padding)
                num_tokens = (tokenized.input_ids != self.tokenizer.pad_token_id).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
        else:
            perplexity = float('inf')
        
        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss if total_tokens > 0 else float('inf'),
            "total_tokens": total_tokens
        }


class MetricsComputer:
    """Computes various NLP metrics for evaluation."""
    
    def __init__(self):
        """Initialize metrics computer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_metric = BLEU()
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        
        Returns:
            Dictionary with ROUGE scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric, score in scores.items():
                rouge_scores[f"{metric}_precision"].append(score.precision)
                rouge_scores[f"{metric}_recall"].append(score.recall)
                rouge_scores[f"{metric}_fmeasure"].append(score.fmeasure)
        
        # Average scores
        avg_scores = {}
        for metric, scores in rouge_scores.items():
            avg_scores[metric] = np.mean(scores)
        
        return avg_scores
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be multiple references per prediction)
        
        Returns:
            Dictionary with BLEU score
        """
        # Convert single references to list format if needed
        if len(references) > 0 and isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        # Compute BLEU
        bleu_score = self.bleu_metric.corpus_score(predictions, list(zip(*references)))
        
        return {
            "bleu": bleu_score.score
        }
    
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        ignore_case: bool = True,
        ignore_punctuation: bool = True
    ) -> Dict[str, float]:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            ignore_case: Whether to ignore case differences
            ignore_punctuation: Whether to ignore punctuation
        
        Returns:
            Dictionary with exact match score
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if ignore_case:
                text = text.lower()
            if ignore_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        
        matches = 0
        for pred, ref in zip(predictions, references):
            pred_norm = normalize_text(pred)
            ref_norm = normalize_text(ref)
            if pred_norm == ref_norm:
                matches += 1
        
        accuracy = matches / len(predictions) if predictions else 0.0
        
        return {
            "exact_match": accuracy,
            "total_examples": len(predictions),
            "correct_examples": matches
        }
    
    def compute_classification_metrics(
        self,
        predictions: List[str],
        references: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: List of predicted labels
            references: List of true labels
            labels: List of possible labels (optional)
        
        Returns:
            Dictionary with classification metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        # Calculate accuracy
        accuracy = accuracy_score(references, predictions)
        
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            references, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Per-class metrics if labels provided
        if labels:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                references, predictions, average=None, labels=labels, zero_division=0
            )
            
            for i, label in enumerate(labels):
                metrics[f"precision_{label}"] = precision_per_class[i] if i < len(precision_per_class) else 0.0
                metrics[f"recall_{label}"] = recall_per_class[i] if i < len(recall_per_class) else 0.0
                metrics[f"f1_{label}"] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        return metrics
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = "generation",
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute all applicable metrics based on task type.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            task_type: Type of task ("generation", "classification", "summarization")
            labels: List of possible labels for classification
        
        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}
        
        if task_type in ["generation", "summarization"]:
            # Text generation metrics
            rouge_metrics = self.compute_rouge(predictions, references)
            all_metrics.update(rouge_metrics)
            
            try:
                bleu_metrics = self.compute_bleu(predictions, [[ref] for ref in references])
                all_metrics.update(bleu_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute BLEU: {e}")
            
            exact_match_metrics = self.compute_exact_match(predictions, references)
            all_metrics.update(exact_match_metrics)
        
        elif task_type == "classification":
            # Classification metrics
            classification_metrics = self.compute_classification_metrics(predictions, references, labels)
            all_metrics.update(classification_metrics)
        
        return all_metrics


def evaluate_model(
    model,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    task_type: str = "generation",
    output_dir: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    compute_perplexity: bool = True,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        task_type: Type of task
        output_dir: Directory to save evaluation results
        generation_config: Configuration for text generation
        compute_perplexity: Whether to compute perplexity
        sample_size: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Starting model evaluation on {len(eval_dataset)} samples")
    
    # Sample dataset if requested
    if sample_size and len(eval_dataset) > sample_size:
        indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
        eval_dataset = eval_dataset.select(indices)
        logger.info(f"Sampled {sample_size} examples for evaluation")
    
    # Initialize evaluators
    text_evaluator = TextGenerationEvaluator(model, tokenizer)
    metrics_computer = MetricsComputer()
    
    evaluation_results = {
        "num_samples": len(eval_dataset),
        "task_type": task_type
    }
    
    # Compute perplexity if requested
    if compute_perplexity:
        perplexity_results = text_evaluator.evaluate_perplexity(eval_dataset)
        evaluation_results.update(perplexity_results)
    
    # Generate text and compute metrics if we have labels
    if "label" in eval_dataset.column_names or "target" in eval_dataset.column_names:
        # Get inputs and references
        inputs = eval_dataset["text"] if "text" in eval_dataset.column_names else eval_dataset["input"]
        references = eval_dataset.get("label", eval_dataset.get("target", []))
        
        if inputs and references:
            # Set generation parameters
            gen_config = generation_config or {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9
            }
            
            # Generate predictions
            logger.info("Generating predictions...")
            predictions = text_evaluator.generate_text(inputs, **gen_config)
            
            # Compute metrics
            logger.info("Computing evaluation metrics...")
            metrics = metrics_computer.compute_all_metrics(
                predictions, references, task_type
            )
            evaluation_results["metrics"] = metrics
            
            # Save sample predictions
            sample_predictions = []
            for i, (input_text, pred, ref) in enumerate(zip(inputs[:10], predictions[:10], references[:10])):
                sample_predictions.append({
                    "input": input_text,
                    "prediction": pred,
                    "reference": ref
                })
            
            evaluation_results["sample_predictions"] = sample_predictions
    
    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_file}")
    
    return evaluation_results


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare evaluation results from multiple models.
    
    Args:
        model_results: Dictionary mapping model names to evaluation results
        output_file: File to save comparison results
    
    Returns:
        Dictionary with comparison results
    """
    if len(model_results) < 2:
        raise ValueError("Need at least 2 models to compare")
    
    comparison_results = {
        "models": list(model_results.keys()),
        "comparison_metrics": {}
    }
    
    # Get common metrics
    common_metrics = None
    for model_name, results in model_results.items():
        metrics = results.get("metrics", {})
        if common_metrics is None:
            common_metrics = set(metrics.keys())
        else:
            common_metrics = common_metrics.intersection(set(metrics.keys()))
    
    # Compare metrics
    for metric in common_metrics:
        metric_values = {}
        for model_name, results in model_results.items():
            metric_values[model_name] = results["metrics"][metric]
        
        # Find best and worst
        if metric in ["accuracy", "f1", "precision", "recall", "bleu", "exact_match"]:
            # Higher is better
            best_model = max(metric_values, key=metric_values.get)
            worst_model = min(metric_values, key=metric_values.get)
        else:
            # Lower is better (loss, perplexity)
            best_model = min(metric_values, key=metric_values.get)
            worst_model = max(metric_values, key=metric_values.get)
        
        comparison_results["comparison_metrics"][metric] = {
            "values": metric_values,
            "best_model": best_model,
            "worst_model": worst_model,
            "best_value": metric_values[best_model],
            "worst_value": metric_values[worst_model]
        }
    
    # Save comparison results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"Model comparison saved to {output_file}")
    
    return comparison_results