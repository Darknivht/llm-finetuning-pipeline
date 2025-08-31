#!/usr/bin/env python3
"""
Simple test script to debug generation issues.
Tests model loading and basic text generation.
"""

import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

def test_basic_generation():
    """Test basic generation with a simple model."""
    print("🧪 Testing Basic Text Generation")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        
        # Test with a small, reliable model
        model_name = "distilgpt2"
        print(f"📦 Loading model: {model_name}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # Load model
        print("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
        
        print(f"✅ Model loaded to {device}")
        print(f"   Parameters: {model.num_parameters():,}")
        
        # Test generation
        print("\n🔮 Testing Generation")
        print("-" * 30)
        
        test_prompts = [
            "The future of artificial intelligence is",
            "Python programming is",
            "The benefits of machine learning include"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)
            
            print(f"Input tokens: {inputs.input_ids.shape[1]}")
            
            # Generate
            with torch.no_grad():
                try:
                    generation_config = GenerationConfig(
                        max_new_tokens=50,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
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
                    
                    print(f"Generated: {output_text.strip()}")
                    print(f"Length: {len(generated_tokens)} tokens")
                    
                except Exception as gen_e:
                    print(f"❌ Generation failed: {gen_e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n✅ Basic generation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backend_loading():
    """Test loading with our backend utilities."""
    print("\n🔧 Testing Backend Model Loading")
    print("=" * 50)
    
    try:
        from backend.models import ModelLoader
        
        model_name = "distilgpt2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading tokenizer via backend...")
        tokenizer = ModelLoader.load_tokenizer(model_name)
        print(f"✅ Backend tokenizer loaded")
        
        print(f"Loading model via backend...")
        model = ModelLoader.load_base_model(model_name, device=device)
        print(f"✅ Backend model loaded")
        
        # Test generation
        test_prompt = "Hello, this is a test of"
        
        print(f"\nTesting generation with prompt: '{test_prompt}'")
        
        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Generated: {output_text.strip()}")
        print("✅ Backend generation test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing Environment")
    print("=" * 30)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Check transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
    
    # Check datasets
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("⚠️ Datasets not available")
    
    # Check PEFT
    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        print("⚠️ PEFT not available")
    
    return True


def main():
    """Run all tests."""
    print("🚀 LLM Pipeline Generation Test")
    print("=" * 60)
    
    # Test environment
    test_environment()
    
    # Test basic generation
    basic_success = test_basic_generation()
    
    # Test backend loading
    backend_success = test_backend_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Basic Generation: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Backend Loading:  {'✅ PASSED' if backend_success else '❌ FAILED'}")
    
    if basic_success and backend_success:
        print("\n🎉 All tests passed! Generation should work.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure transformers>=4.30.0 is installed")
        print("2. Check internet connection for model downloads")
        print("3. Verify CUDA installation if using GPU")
        print("4. Try running: pip install --upgrade transformers torch")
    
    return basic_success and backend_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)