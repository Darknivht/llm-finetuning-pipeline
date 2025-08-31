#!/usr/bin/env python3
"""
Basic training example for the LLM Fine-Tuning Pipeline.
Shows how to perform simple LoRA fine-tuning with synthetic data.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend import Config, setup_logging
from backend.models import ModelLoader
from backend.data_utils import DataProcessor, generate_synthetic_data
from backend.train_utils import create_training_arguments, setup_trainer, save_model_with_config

def main():
    """Basic training example."""
    print("🚀 Basic Training Example")
    print("=" * 40)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Configuration
    model_name = "distilgpt2"
    output_dir = Path("./examples/outputs/basic_model")
    data_dir = Path("./examples/data")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Data directory: {data_dir}")
    
    # Step 1: Generate synthetic data
    print("\n📝 Step 1: Generating synthetic data...")
    
    data_file = data_dir / "synthetic_data.csv"
    generate_synthetic_data(
        output_file=data_file,
        num_samples=200,
        task_type="completion",
        format_type="csv"
    )
    print(f"✅ Generated {data_file}")
    
    # Step 2: Load tokenizer
    print(f"\n🔤 Step 2: Loading tokenizer ({model_name})...")
    tokenizer = ModelLoader.load_tokenizer(model_name)
    print("✅ Tokenizer loaded")
    
    # Step 3: Prepare dataset
    print("\n📊 Step 3: Preparing dataset...")
    
    data_processor = DataProcessor(tokenizer=tokenizer, max_length=256)
    dataset_dict = data_processor.prepare_dataset(
        file_path=data_file,
        output_dir=data_dir / "processed",
        text_column="text",
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        clean_data=True,
        max_examples=200
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
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    )
    print("✅ PEFT model configured")
    
    # Step 6: Create training arguments
    print("\n🏃 Step 6: Setting up training...")
    
    training_args = create_training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        fp16=False,  # Disable for compatibility
        load_best_model_at_end=True
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
    print("This may take a few minutes...")
    
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Step 9: Save the model
        print("\n💾 Step 9: Saving model...")
        
        final_model_dir = output_dir / "final"
        training_config = {
            "model_name_or_path": model_name,
            "use_peft": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "max_length": 256,
            "training_samples": len(train_dataset),
            "validation_samples": len(eval_dataset),
            "epochs": 2
        }
        
        save_model_with_config(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(final_model_dir),
            config_dict=training_config
        )
        
        print(f"✅ Model saved to {final_model_dir}")
        
        # Step 10: Test inference
        print("\n🔮 Step 10: Testing inference...")
        
        import torch
        
        test_prompt = "The benefits of artificial intelligence include"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"Input: {test_prompt}")
        print(f"Output: {generated_text}")
        print("✅ Inference test completed!")
        
        # Summary
        print("\n🎉 Training Example Completed!")
        print("=" * 40)
        print(f"📁 Model saved to: {final_model_dir}")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(eval_dataset)}")
        print("\nNext steps:")
        print(f"1. Test inference: python inference.py --checkpoint {final_model_dir}")
        print(f"2. Run evaluation: python evaluate.py --checkpoint {final_model_dir} --dataset {data_dir / 'processed'}")
        print("3. Launch demo: streamlit run demo_streamlit.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)