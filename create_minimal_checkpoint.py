#!/usr/bin/env python3
"""
Create a minimal working checkpoint for testing generation without downloading large models.
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

def create_minimal_checkpoint():
    """Create a minimal checkpoint using the smallest possible model."""
    print("🏗️ Creating Minimal Test Checkpoint")
    print("=" * 40)
    
    try:
        from backend.models import ModelLoader
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Create checkpoints directory
        checkpoints_dir = Path("c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Use the distilgpt2 model (smallest GPT model)
        model_name = "distilgpt2"
        device = "cpu"
        
        print(f"Using {model_name} on {device}")
        
        # Load directly with transformers to avoid large downloads if possible
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Tokenizer loaded")
        
        # Create checkpoint directory
        checkpoint_name = "minimal-test"
        checkpoint_path = checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save tokenizer first
        tokenizer.save_pretrained(str(checkpoint_path))
        print(f"✅ Tokenizer saved to {checkpoint_path}")
        
        # Try to load model in the most efficient way
        print("Loading model (this may take a few minutes for first download)...")
        
        # Use smaller precision and minimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        print("✅ Model loaded")
        
        # Save model
        model.save_pretrained(str(checkpoint_path))
        print(f"✅ Model saved to {checkpoint_path}")
        
        # Create training config
        config = {
            "base_model": model_name,
            "model_name": model_name,
            "use_peft": False,
            "device": device,
            "model_type": "causal_lm",
            "checkpoint_name": checkpoint_name,
            "created_for_testing": True,
            "max_length": 512,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "architecture": "GPTNeoXForCausalLM" if "neox" in model_name.lower() else "GPT2LMHeadModel"
        }
        
        config_file = checkpoint_path / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        
        return checkpoint_path, config
        
    except Exception as e:
        print(f"❌ Failed to create minimal checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def quick_test_generation(checkpoint_path, config):
    """Quick test of generation capability."""
    print(f"\n🧪 Testing Generation with {checkpoint_path.name}")
    print("=" * 50)
    
    try:
        from backend.models import ModelLoader
        import torch
        from transformers import GenerationConfig
        
        device = config.get("device", "cpu")
        
        # Load checkpoint
        model, tokenizer, loaded_config = ModelLoader.load_checkpoint(
            checkpoint_path, device=device
        )
        
        print("✅ Checkpoint loaded successfully")
        
        # Simple test
        prompt = "Hello, how are you"
        print(f"Testing with prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=100).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"✅ Generated: '{generated_text.strip()}'")
        
        if generated_text.strip():
            print("🎉 Generation successful!")
            return True
        else:
            print("⚠️ Empty generation, but no errors")
            return True
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🚀 Creating Minimal Working Checkpoint")
    print("=" * 50)
    
    checkpoint_path, config = create_minimal_checkpoint()
    
    if not checkpoint_path:
        print("❌ Could not create checkpoint")
        return False
    
    # Test generation
    success = quick_test_generation(checkpoint_path, config)
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print("=" * 50)
        print(f"✅ Minimal checkpoint created: {checkpoint_path.name}")
        print("✅ Generation tested and working")
        print("\n📋 Next Steps:")
        print("1. Restart Streamlit demo if running")
        print("2. Select 'Local Model' in the demo")
        print(f"3. Choose '{checkpoint_path.name}' from the dropdown")
        print("4. Test text generation")
        return True
    else:
        print("\n❌ Generation test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)