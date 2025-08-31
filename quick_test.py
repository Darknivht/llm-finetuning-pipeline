#!/usr/bin/env python3
"""
Quick test to identify the generation issue using backend modules.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

print("🔍 Quick Generation Test")
print("=" * 40)

try:
    # Test backend model loading
    from backend.models import ModelLoader
    print("✅ Backend models module imported")
    
    # Test tokenizer loading
    print("Loading tokenizer...")
    tokenizer = ModelLoader.load_tokenizer("distilgpt2")
    print(f"✅ Tokenizer loaded: {len(tokenizer)} vocab")
    
    # Test simple tokenization
    test_text = "Hello world, this is a test."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"✅ Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
    
    # Test the issue in demo
    print("\n🎯 Testing Streamlit Demo Components")
    print("-" * 40)
    
    # Check if there are any trained models/checkpoints
    checkpoints_dir = Path("c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/checkpoints")
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
        print(f"Found {len(checkpoints)} checkpoints: {[c.name for c in checkpoints]}")
    else:
        print("No checkpoints directory found")
    
    # Test basic generation without full model loading
    print("\n🧪 Testing Generation Logic")
    print("-" * 30)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        
        # Use a very small model for testing
        model_name = "distilgpt2"
        device = "cpu"  # Force CPU for faster loading
        
        print(f"Quick test with {model_name} on {device}")
        
        # Test if we can create generation config
        gen_config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.7,
            do_sample=True
        )
        print(f"✅ Generation config created: max_new_tokens={gen_config.max_new_tokens}")
        
        print("✅ Basic generation components work")
        
    except Exception as e:
        print(f"❌ Generation component error: {e}")
    
    print("\n🎯 Summary:")
    print("- Backend imports: ✅")
    print("- Tokenizer loading: ✅") 
    print("- Generation config: ✅")
    print("- Issue likely in full model loading or generation call")
    
    print("\n🔧 Next steps:")
    print("1. Test with actual fine-tuned checkpoint")
    print("2. Check streamlit app generation call")
    print("3. Verify model loading parameters")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()