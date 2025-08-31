#!/usr/bin/env python3
"""
Custom data training example for the LLM Fine-Tuning Pipeline.
Shows how to fine-tune with your own CSV data.
"""

import sys
import csv
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend import setup_logging
from backend.models import ModelLoader
from backend.data_utils import DataProcessor
from backend.train_utils import create_training_arguments, setup_trainer, save_model_with_config


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_data = [
        {"text": "Python is a programming language", "category": "programming"},
        {"text": "Machine learning helps computers learn", "category": "ai"},
        {"text": "Data science involves analyzing data", "category": "data"},
        {"text": "Neural networks mimic brain functions", "category": "ai"},
        {"text": "JavaScript runs in web browsers", "category": "programming"},
        {"text": "Statistics help understand data patterns", "category": "data"},
        {"text": "Deep learning uses neural networks", "category": "ai"},
        {"text": "HTML structures web content", "category": "programming"},
        {"text": "Visualization makes data understandable", "category": "data"},
        {"text": "Natural language processing understands text", "category": "ai"},
        {"text": "CSS styles web pages beautifully", "category": "programming"},
        {"text": "Regression models predict continuous values", "category": "data"},
        {"text": "Computer vision recognizes images", "category": "ai"},
        {"text": "SQL queries database information", "category": "programming"},
        {"text": "Classification categorizes data points", "category": "data"},
        {"text": "Transformers revolutionized NLP tasks", "category": "ai"},
        {"text": "Git tracks code changes", "category": "programming"},
        {"text": "Correlation measures variable relationships", "category": "data"},
        {"text": "GANs generate realistic fake data", "category": "ai"},
        {"text": "Docker containerizes applications", "category": "programming"}
    ]
    
    return sample_data


def prepare_custom_data(data_file: Path, output_dir: Path):
    """Prepare custom data for training."""
    print(f"📝 Creating sample dataset: {data_file}")
    
    # Create sample data if file doesn't exist
    if not data_file.exists():
        sample_data = create_sample_dataset()
        
        with open(data_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'category'])
            writer.writeheader()
            writer.writerows(sample_data)
        
        print(f"✅ Created sample dataset with {len(sample_data)} examples")
    else:
        print(f"✅ Using existing dataset: {data_file}")
    
    return data_file


def main():
    """Custom data training example."""
    print("🚀 Custom Data Training Example")
    print("=" * 45)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Configuration
    model_name = "distilgpt2"
    output_dir = Path("./examples/outputs/custom_model")
    data_dir = Path("./examples/data")
    data_file = data_dir / "custom_dataset.csv"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Data directory: {data_dir}")
    
    # Step 1: Prepare custom data
    print("\n📝 Step 1: Preparing custom data...")
    prepare_custom_data(data_file, data_dir)
    
    # Step 2: Load tokenizer
    print(f"\n🔤 Step 2: Loading tokenizer ({model_name})...")
    tokenizer = ModelLoader.load_tokenizer(model_name)
    print("✅ Tokenizer loaded")
    
    # Step 3: Process the dataset
    print("\n📊 Step 3: Processing dataset...")
    
    data_processor = DataProcessor(tokenizer=tokenizer, max_length=128)
    
    # For this example, we'll create text completion tasks from the data
    # Format: "Category: [category]. Description: [text]"
    
    print("📋 Transforming data for text completion...")
    
    # Read and transform the data
    transformed_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create completion format
            completion_text = f"Category: {row['category']}. Description: {row['text']}"
            transformed_data.append({"text": completion_text})
    
    # Save transformed data
    transformed_file = data_dir / "transformed_data.csv"
    with open(transformed_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text'])
        writer.writeheader()
        writer.writerows(transformed_data)
    
    print(f"✅ Transformed {len(transformed_data)} examples")
    
    # Prepare dataset for training
    dataset_dict = data_processor.prepare_dataset(
        file_path=transformed_file,
        output_dir=data_dir / "processed",
        text_column="text",
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        clean_data=True
    )
    
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(eval_dataset)}")
    
    # Step 4: Load base model
    print(f"\n🤖 Step 4: Loading base model ({model_name})...")
    model = ModelLoader.load_base_model(model_name, device="auto")
    print("✅ Base model loaded")
    
    # Step 5: Setup PEFT
    print("\n⚡ Step 5: Setting up PEFT/LoRA...")
    model = ModelLoader.setup_peft_model(
        model,
        lora_rank=8,  # Smaller rank for small dataset
        lora_alpha=16,
        lora_dropout=0.1
    )
    print("✅ PEFT model configured")
    
    # Step 6: Create training arguments
    print("\n🏃 Step 6: Setting up training...")
    
    training_args = create_training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=5,  # More epochs for small dataset
        per_device_train_batch_size=2,  # Smaller batch size
        per_device_eval_batch_size=2,
        learning_rate=3e-4,  # Higher learning rate for small dataset
        warmup_steps=20,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Step 7: Setup trainer
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args
    )
    
    print("✅ Trainer configured")
    
    # Step 8: Train the model
    print("\n🚂 Step 8: Training model...")
    print("Training with custom data (this may take a few minutes)...")
    
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Step 9: Save the model
        print("\n💾 Step 9: Saving model...")
        
        final_model_dir = output_dir / "final"
        training_config = {
            "model_name_or_path": model_name,
            "use_peft": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "max_length": 128,
            "training_samples": len(train_dataset),
            "validation_samples": len(eval_dataset),
            "epochs": 5,
            "data_format": "custom_csv",
            "task_type": "text_completion"
        }
        
        save_model_with_config(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(final_model_dir),
            config_dict=training_config
        )
        
        print(f"✅ Model saved to {final_model_dir}")
        
        # Step 10: Test with domain-specific prompts
        print("\n🔮 Step 10: Testing with domain prompts...")
        
        import torch
        
        test_prompts = [
            "Category: programming. Description:",
            "Category: ai. Description:",
            "Category: data. Description:"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text.strip()}")
        
        print("\n✅ Domain-specific inference tests completed!")
        
        # Summary and next steps
        print("\n🎉 Custom Data Training Completed!")
        print("=" * 45)
        print(f"📁 Model saved to: {final_model_dir}")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(eval_dataset)}")
        print(f"📝 Data format: Custom CSV with categories")
        print("\nWhat you learned:")
        print("✅ How to prepare custom CSV data")
        print("✅ How to transform data for specific tasks")
        print("✅ How to adjust hyperparameters for small datasets")
        print("✅ How to test domain-specific generation")
        
        print("\nNext steps:")
        print(f"1. Interactive testing: python inference.py --checkpoint {final_model_dir} --interactive")
        print(f"2. Batch inference: python inference.py --checkpoint {final_model_dir} --input_file prompts.txt")
        print("3. Create more data and retrain for better results")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)