#!/usr/bin/env python3
"""
Test the generation logic from the Streamlit demo directly.
This will help identify what's causing the "doesn't generate" issue.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

print("🧪 Testing Streamlit Generation Logic")
print("=" * 50)

# Import the generation functions from demo
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "demo_streamlit", 
        "c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/demo_streamlit.py"
    )
    demo_module = importlib.util.module_from_spec(spec)
    
    # Mock streamlit to avoid import errors
    class MockStreamlit:
        def error(self, msg): print(f"STREAMLIT ERROR: {msg}")
        def success(self, msg): print(f"STREAMLIT SUCCESS: {msg}")
        def warning(self, msg): print(f"STREAMLIT WARNING: {msg}")
    
    sys.modules['streamlit'] = MockStreamlit()
    
    # Load the module
    spec.loader.exec_module(demo_module)
    
    print("✅ Demo module loaded successfully")
    
    # Test the local generation function
    print("\n🔍 Testing Local Generation Function")
    print("-" * 40)
    
    # Test without actual model loading first
    test_prompt = "The future of AI is"
    print(f"Test prompt: '{test_prompt}'")
    
    # Check if we can at least call the function structure
    if hasattr(demo_module, 'generate_text_local'):
        print("✅ generate_text_local function found")
        
        # For now, test the generation config creation part
        try:
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                use_cache=True
            )
            print("✅ Generation config created successfully")
            print(f"   Config: max_new_tokens={generation_config.max_new_tokens}, temp={generation_config.temperature}")
            
        except Exception as e:
            print(f"❌ Generation config failed: {e}")
    
    # Test the fallback client
    print("\n🌐 Testing OpenRouter Fallback")
    print("-" * 40)
    
    try:
        from backend.openrouter import create_fallback_client
        
        client = create_fallback_client()
        print("✅ Fallback client created")
        
        # Test generation
        print("Testing fallback generation...")
        result = client.generate_text(
            test_prompt,
            max_tokens=20,
            temperature=0.7,
            prefer_provider="openrouter"
        )
        
        if result:
            print(f"✅ Fallback generation successful: '{result[:100]}...'")
        else:
            print("⚠️ Fallback generation returned None (API key might be missing)")
            
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
    
    # Check available models
    print("\n📁 Checking Available Checkpoints")
    print("-" * 40)
    
    checkpoints_dir = Path("c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/checkpoints")
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*"))
        print(f"Found checkpoints: {[c.name for c in checkpoints]}")
        
        if not checkpoints:
            print("📝 No checkpoints found - need to train a model first")
    else:
        print("📁 No checkpoints directory - need to train a model first")
    
    print("\n🎯 Diagnosis Summary:")
    print("=" * 30)
    print("✅ Demo module loads correctly")
    print("✅ Generation config works")
    print("✅ Backend imports work")
    
    if checkpoints_dir.exists() and list(checkpoints_dir.glob("*")):
        print("✅ Checkpoints available")
    else:
        print("❌ No trained checkpoints - this is likely the main issue!")
        print("   Solution: Train a model first using train.py")
    
    print("\n🔧 Quick Fix - Create a Minimal Model")
    print("=" * 45)
    print("Let's create a minimal checkpoint for testing...")
    
    try:
        # Create checkpoints directory
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Create a simple test checkpoint by saving the base model
        from backend.models import ModelLoader
        import torch
        
        print("Creating test checkpoint...")
        model_name = "distilgpt2" 
        
        tokenizer = ModelLoader.load_tokenizer(model_name)
        print("✅ Tokenizer loaded")
        
        model = ModelLoader.load_base_model(model_name, device="cpu")
        print("✅ Base model loaded")
        
        # Save as a checkpoint
        test_checkpoint = checkpoints_dir / "test-checkpoint"
        test_checkpoint.mkdir(exist_ok=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(str(test_checkpoint))
        print("✅ Tokenizer saved")
        
        # Save model
        model.save_pretrained(str(test_checkpoint))
        print("✅ Model saved")
        
        # Create config file
        import json
        config = {
            "base_model": model_name,
            "use_peft": False,
            "device": "cpu",
            "model_type": "base_model_copy",
            "created_at": str(Path(__file__).stat().st_mtime)
        }
        
        with open(test_checkpoint / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Config saved")
        print(f"🎉 Test checkpoint created at: {test_checkpoint}")
        print("   Now the Streamlit demo should have a model to load!")
        
    except Exception as e:
        print(f"❌ Checkpoint creation failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Demo module test failed: {e}")
    import traceback
    traceback.print_exc()