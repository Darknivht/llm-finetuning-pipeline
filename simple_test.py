#!/usr/bin/env python3
"""
Simple test to check Python environment and package availability.
"""

print("🔍 Checking Python Environment")
print("=" * 40)

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test basic imports
packages = [
    "torch",
    "transformers", 
    "datasets",
    "peft",
    "streamlit",
    "openai",
    "sklearn"
]

print("\n📦 Testing Package Imports:")
print("-" * 30)

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"❌ {package}: {e}")

print("\n🔧 Testing Torch Specifically:")
print("-" * 30)

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

print("\n🤖 Testing Transformers:")
print("-" * 30)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers classes imported successfully")
    
    # Test with a tiny model
    print("Testing tokenizer loading...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"✅ GPT-2 tokenizer loaded, vocab size: {len(tokenizer)}")
    
except Exception as e:
    print(f"❌ Transformers error: {e}")

print("\n🎯 Environment Check Complete!")