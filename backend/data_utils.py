"""
Data processing utilities for the LLM Fine-Tuning Pipeline.
Supports CSV, JSON, and TSV formats with flexible column mapping.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .logging_config import get_logger

logger = get_logger("data_utils")


class DataProcessor:
    """Main class for data processing and dataset preparation."""
    
    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Initialize data processor.
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def load_data_file(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = None,
        encoding: str = "utf-8"
    ) -> List[Dict[str, str]]:
        """
        Load data from CSV, JSON, or TSV file.
        
        Args:
            file_path: Path to data file
            text_column: Name of text column
            label_column: Name of label column (optional)
            encoding: File encoding
        
        Returns:
            List of data samples
        """
        file_path = Path(file_path)
        logger.info(f"Loading data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = []
        
        if file_path.suffix.lower() == '.csv':
            data = self._load_csv(file_path, text_column, label_column, encoding)
        elif file_path.suffix.lower() == '.tsv':
            data = self._load_tsv(file_path, text_column, label_column, encoding)
        elif file_path.suffix.lower() == '.json':
            data = self._load_json(file_path, text_column, label_column, encoding)
        elif file_path.suffix.lower() == '.jsonl':
            data = self._load_jsonl(file_path, text_column, label_column, encoding)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def _load_csv(
        self, 
        file_path: Path, 
        text_column: str, 
        label_column: Optional[str], 
        encoding: str
    ) -> List[Dict[str, str]]:
        """Load data from CSV file."""
        data = []
        with open(file_path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column not in row:
                    raise ValueError(f"Text column '{text_column}' not found in CSV")
                
                sample = {"text": row[text_column].strip()}
                if label_column and label_column in row:
                    sample["label"] = row[label_column].strip()
                
                data.append(sample)
        return data
    
    def _load_tsv(
        self, 
        file_path: Path, 
        text_column: str, 
        label_column: Optional[str], 
        encoding: str
    ) -> List[Dict[str, str]]:
        """Load data from TSV file."""
        data = []
        with open(file_path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if text_column not in row:
                    raise ValueError(f"Text column '{text_column}' not found in TSV")
                
                sample = {"text": row[text_column].strip()}
                if label_column and label_column in row:
                    sample["label"] = row[label_column].strip()
                
                data.append(sample)
        return data
    
    def _load_json(
        self, 
        file_path: Path, 
        text_column: str, 
        label_column: Optional[str], 
        encoding: str
    ) -> List[Dict[str, str]]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding=encoding) as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        data = []
        for item in raw_data:
            if text_column not in item:
                raise ValueError(f"Text column '{text_column}' not found in JSON item")
            
            sample = {"text": str(item[text_column]).strip()}
            if label_column and label_column in item:
                sample["label"] = str(item[label_column]).strip()
            
            data.append(sample)
        return data
    
    def _load_jsonl(
        self, 
        file_path: Path, 
        text_column: str, 
        label_column: Optional[str], 
        encoding: str
    ) -> List[Dict[str, str]]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    if text_column not in item:
                        raise ValueError(f"Text column '{text_column}' not found in line {line_num}")
                    
                    sample = {"text": str(item[text_column]).strip()}
                    if label_column and label_column in item:
                        sample["label"] = str(item[label_column]).strip()
                    
                    data.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON on line {line_num}: {e}")
                    continue
        return data
    
    def clean_text(
        self, 
        text: str, 
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> str:
        """
        Clean text data.
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            remove_extra_whitespace: Whether to remove extra whitespace
            min_length: Minimum text length
            max_length: Maximum text length
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = ' '.join(text.split())
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Check length constraints
        if len(text) < min_length:
            return ""
        
        if max_length and len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return text.strip()
    
    def split_data(
        self,
        data: List[Dict[str, str]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Shuffle data
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        n_samples = len(data_copy)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data = data_copy[:n_train]
        val_data = data_copy[n_train:n_train + n_val]
        test_data = data_copy[n_train + n_val:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data
    
    def create_dataset(
        self,
        data: List[Dict[str, str]],
        tokenize: bool = True
    ) -> Dataset:
        """
        Create HuggingFace Dataset from data list.
        
        Args:
            data: Input data
            tokenize: Whether to tokenize texts
        
        Returns:
            HuggingFace Dataset
        """
        if not data:
            raise ValueError("Cannot create dataset from empty data")
        
        # Convert to Dataset
        dataset = Dataset.from_list(data)
        
        # Apply tokenization if requested and tokenizer is available
        if tokenize and self.tokenizer is not None:
            logger.info("Tokenizing dataset...")
            dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )
        
        return dataset
    
    def _tokenize_function(self, examples):
        """Tokenization function for dataset mapping."""
        # Handle both single text and text-label pairs
        if "label" in examples:
            # For supervised fine-tuning: input + target
            inputs = examples["text"]
            targets = examples["label"]
            
            # Tokenize inputs and targets separately
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation
            )
            
            target_encodings = self.tokenizer(
                targets,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation
            )
            
            # Set labels for loss computation
            model_inputs["labels"] = target_encodings["input_ids"]
        else:
            # For unsupervised/language modeling: text as both input and target
            model_inputs = self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation
            )
            
            # For causal LM, labels are the same as input_ids (shifted internally by model)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def prepare_dataset(
        self,
        file_path: Union[str, Path],
        output_dir: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        clean_data: bool = True,
        max_examples: Optional[int] = None,
        random_seed: int = 42
    ) -> DatasetDict:
        """
        Complete data preparation pipeline.
        
        Args:
            file_path: Input data file path
            output_dir: Output directory for processed datasets
            text_column: Name of text column
            label_column: Name of label column
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            clean_data: Whether to clean text data
            max_examples: Maximum number of examples to use
            random_seed: Random seed
        
        Returns:
            DatasetDict with train/val/test splits
        """
        # Load raw data
        raw_data = self.load_data_file(file_path, text_column, label_column)
        
        # Limit examples if specified
        if max_examples and len(raw_data) > max_examples:
            random.seed(random_seed)
            raw_data = random.sample(raw_data, max_examples)
            logger.info(f"Limited dataset to {max_examples} examples")
        
        # Clean data if requested
        if clean_data:
            logger.info("Cleaning text data...")
            cleaned_data = []
            for item in raw_data:
                cleaned_text = self.clean_text(item["text"])
                if cleaned_text:  # Only keep non-empty texts
                    cleaned_item = {"text": cleaned_text}
                    if "label" in item:
                        cleaned_item["label"] = self.clean_text(item["label"])
                    cleaned_data.append(cleaned_item)
            
            logger.info(f"Cleaned dataset: {len(raw_data)} -> {len(cleaned_data)} samples")
            raw_data = cleaned_data
        
        # Split data
        train_data, val_data, test_data = self.split_data(
            raw_data, train_ratio, val_ratio, test_ratio, random_seed
        )
        
        # Create datasets
        train_dataset = self.create_dataset(train_data)
        val_dataset = self.create_dataset(val_data)
        test_dataset = self.create_dataset(test_data)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        # Save datasets
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_dir))
        logger.info(f"Datasets saved to {output_dir}")
        
        return dataset_dict


def generate_synthetic_data(
    output_file: Union[str, Path],
    num_samples: int = 100,
    task_type: str = "completion",
    format_type: str = "csv"
) -> None:
    """
    Generate synthetic data for testing the pipeline.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to generate
        task_type: Type of task ("completion", "classification", "summarization")
        format_type: Output format ("csv", "json", "jsonl")
    """
    logger.info(f"Generating {num_samples} synthetic samples for {task_type} task")
    
    random.seed(42)
    data = []
    
    if task_type == "completion":
        # Simple text completion
        prompts = [
            "The weather today is",
            "I like to eat",
            "My favorite color is",
            "In the future, I want to",
            "The best way to learn is",
        ]
        
        completions = [
            "sunny and warm",
            "pizza and ice cream",
            "blue like the ocean",
            "travel around the world",
            "by practicing every day",
        ]
        
        for i in range(num_samples):
            prompt = random.choice(prompts)
            completion = random.choice(completions)
            
            data.append({
                "text": f"{prompt}",
                "label": f"{completion}"
            })
    
    elif task_type == "classification":
        # Sentiment classification
        texts = [
            "This movie is amazing!",
            "I hate waiting in long lines.",
            "The food was okay, nothing special.",
            "Best day ever!",
            "This is terrible.",
            "Pretty good experience overall.",
        ]
        
        labels = ["positive", "negative", "neutral"]
        
        for i in range(num_samples):
            text = random.choice(texts)
            label = random.choice(labels)
            
            data.append({
                "text": text,
                "label": label
            })
    
    elif task_type == "summarization":
        # Text summarization
        articles = [
            "Scientists have discovered a new species of fish in the deep ocean. The fish has unique bioluminescent properties.",
            "The company announced record profits this quarter, exceeding analyst expectations by 15%.",
            "Local residents are concerned about the new construction project affecting traffic in the downtown area.",
        ]
        
        summaries = [
            "New bioluminescent fish species discovered",
            "Company reports record quarterly profits",
            "Construction project raises traffic concerns",
        ]
        
        for i in range(num_samples):
            article = random.choice(articles)
            summary = random.choice(summaries)
            
            data.append({
                "text": article,
                "label": summary
            })
    
    # Save data in specified format
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "csv":
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(data)
    
    elif format_type == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format_type == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    logger.info(f"Synthetic data saved to {output_file}")


def load_huggingface_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    text_column: str = "text",
    label_column: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        subset: Dataset subset/configuration
        split: Specific split to load
        text_column: Name of text column
        label_column: Name of label column
        cache_dir: Cache directory for downloaded datasets
    
    Returns:
        DatasetDict with loaded data
    """
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    
    try:
        if split:
            dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
            # Convert single split to DatasetDict
            dataset_dict = DatasetDict({split: dataset})
        else:
            dataset_dict = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        
        # Rename columns if necessary
        for split_name, split_dataset in dataset_dict.items():
            if text_column != "text" and text_column in split_dataset.column_names:
                split_dataset = split_dataset.rename_column(text_column, "text")
            
            if label_column and label_column != "label" and label_column in split_dataset.column_names:
                split_dataset = split_dataset.rename_column(label_column, "label")
            
            dataset_dict[split_name] = split_dataset
        
        logger.info(f"Loaded dataset with splits: {list(dataset_dict.keys())}")
        return dataset_dict
    
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise