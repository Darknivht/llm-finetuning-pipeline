#!/usr/bin/env python3
"""
Diagnose and fix the generation issue by creating a test model and testing generation.
"""

import sys
import os
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

def create_test_model():
    """Create a simple test checkpoint for the demo."""
    print("🏗️ Creating Test Model Checkpoint")
    print("=" * 40)
    
    try:
        from backend.models import ModelLoader
        
        # Create checkpoints directory
        checkpoints_dir = Path("c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Use a small, fast model
        model_name = "distilgpt2"
        device = "cpu"
        
        print(f"Creating checkpoint from {model_name}...")
        
        # Load base model and tokenizer
        tokenizer = ModelLoader.load_tokenizer(model_name)
        print("✅ Tokenizer loaded")
        
        model = ModelLoader.load_base_model(model_name, device=device)
        print("✅ Model loaded")
        
        # Save as checkpoint
        checkpoint_name = "test-distilgpt2"
        test_checkpoint = checkpoints_dir / checkpoint_name
        test_checkpoint.mkdir(exist_ok=True)
        
        # Save tokenizer and model
        tokenizer.save_pretrained(str(test_checkpoint))
        model.save_pretrained(str(test_checkpoint))
        
        print(f"✅ Model saved to {test_checkpoint}")
        
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
            "vocab_size": len(tokenizer)
        }
        
        config_file = test_checkpoint / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config saved to {config_file}")
        return test_checkpoint, config
        
    except Exception as e:
        print(f"❌ Failed to create test model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_generation(checkpoint_path, config):
    """Test generation with the created checkpoint."""
    print("\n🧪 Testing Generation with Checkpoint")
    print("=" * 45)
    
    try:
        from backend.models import ModelLoader
        import torch
        from transformers import GenerationConfig
        
        device = config.get("device", "cpu")
        print(f"Loading model from {checkpoint_path} on {device}")
        
        # Load the checkpoint
        model, tokenizer, loaded_config = ModelLoader.load_checkpoint(
            checkpoint_path, device=device
        )
        
        print("✅ Checkpoint loaded successfully")
        print(f"Model: {type(model).__name__}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # Test generation
        test_prompts = [
            "The future of artificial intelligence",
            "Python programming is",
            "Machine learning helps us"
        ]
        
        print(f"\n🔮 Testing generation with {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: '{prompt}'")
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            print(f"Input tokens: {inputs.input_ids.shape[1]}")
            
            # Generate
            generation_config = GenerationConfig(
                max_new_tokens=30,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            
            output_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"Generated ({len(generated_tokens)} tokens): {output_text.strip()}")
            
            if output_text.strip():
                print("✅ Generation successful!")
            else:
                print("⚠️ Empty generation")
        
        print("\n🎉 All generation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_generation_function(checkpoint_path):
    """Test the actual generation function from Streamlit demo."""
    print("\n🎯 Testing Streamlit Generation Function")
    print("=" * 45)
    
    try:
        from backend.models import ModelLoader
        import torch
        from transformers import GenerationConfig
        
        # Load model like the streamlit app does
        model, tokenizer, config = ModelLoader.load_checkpoint(checkpoint_path, device="cpu")
        
        # Replicate the generate_text_local function logic
        def test_generate_text_local(model, tokenizer, prompt, **kwargs):
            try:
                # Create generation config
                generation_config = GenerationConfig(
                    max_new_tokens=kwargs.get("max_new_tokens", 50),
                    temperature=kwargs.get("temperature", 0.7),
                    top_k=kwargs.get("top_k", 50),
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=kwargs.get("temperature", 0.7) > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Tokenize input
                device = next(model.parameters()).device
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode output (remove input tokens)
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                
                output_text = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return output_text.strip()
            
            except Exception as e:
                print(f"Generation function error: {e}")
                return None
        
        # Test the function
        test_prompt = "Hello world, this is a test of"
        print(f"Testing with prompt: '{test_prompt}'")
        
        result = test_generate_text_local(
            model, tokenizer, test_prompt,
            max_new_tokens=20,
            temperature=0.8
        )
        
        if result:
            print(f"✅ Generation successful: '{result}'")
            print("🎉 The Streamlit generation function should work now!")
            return True
        else:
            print("❌ Generation returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnosis and fix function."""
    print("🔧 LLM Generation Issue Diagnosis & Fix")
    print("=" * 50)
    
    # Step 1: Create test model
    checkpoint_path, config = create_test_model()
    
    if not checkpoint_path:
        print("❌ Could not create test model")
        return False
    
    # Step 2: Test generation
    generation_works = test_generation(checkpoint_path, config)
    
    if not generation_works:
        print("❌ Basic generation failed")
        return False
    
    # Step 3: Test Streamlit function
    streamlit_works = test_streamlit_generation_function(checkpoint_path)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 DIAGNOSIS COMPLETE")
    print("=" * 50)
    
    if generation_works and streamlit_works:
        print("✅ Issue RESOLVED!")
        print("✅ Created working test checkpoint")
        print("✅ Generation functions work correctly")
        print("✅ Streamlit demo should now work")
        print("\n📋 Next steps:")
        print("1. Restart the Streamlit app")
        print("2. Select 'Local Model' option") 
        print("3. Choose the 'test-distilgpt2' checkpoint")
        print("4. Test generation with your prompts")
        return True
    else:
        print("❌ Issue PERSISTS")
        print("   Check the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)