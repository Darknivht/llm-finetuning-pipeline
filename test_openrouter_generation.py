#!/usr/bin/env python3
"""
Test OpenRouter generation directly to verify the fallback works.
This should solve the "doesn't generate" issue immediately.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

def test_openrouter_generation():
    """Test OpenRouter generation capability."""
    print("🌐 Testing OpenRouter Generation")
    print("=" * 40)
    
    try:
        from backend.openrouter import create_fallback_client
        
        # Create the client
        client = create_fallback_client()
        print("✅ OpenRouter client created")
        
        # Test prompts
        test_prompts = [
            "Complete this sentence: The future of AI is",
            "Write a short poem about technology:",
            "Explain machine learning in simple terms:"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}: '{prompt}'")
            
            try:
                result = client.generate_text(
                    prompt,
                    max_tokens=50,
                    temperature=0.7,
                    prefer_provider="openrouter"
                )
                
                if result:
                    print(f"✅ Generated: '{result[:100]}...'")
                else:
                    print("⚠️ No result returned (might need API key)")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_env_config():
    """Check .env configuration for API keys."""
    print("\n🔍 Checking Environment Configuration")
    print("=" * 45)
    
    env_file = Path("c:/Users/Toshiba/Desktop/llm-finetuning-pipeline/.env")
    
    if not env_file.exists():
        print("❌ No .env file found")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        print("✅ .env file found")
        
        # Check for OpenRouter key
        if "OPENROUTER_API_KEY=" in content:
            lines = content.split('\n')
            for line in lines:
                if line.startswith("OPENROUTER_API_KEY="):
                    key_value = line.split('=', 1)[1].strip()
                    if key_value and key_value != "":
                        print("✅ OpenRouter API key is set")
                        print(f"   Key: {key_value[:10]}...{key_value[-4:] if len(key_value) > 14 else 'short'}")
                        return True
                    else:
                        print("❌ OpenRouter API key is empty")
                        return False
        else:
            print("❌ No OpenRouter API key configured")
            return False
            
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def create_simple_generation_test():
    """Create a simple test that works immediately."""
    print("\n🚀 Creating Direct Generation Test")
    print("=" * 40)
    
    try:
        # Test the basic generation logic that Streamlit uses
        from backend.openrouter import create_fallback_client
        
        client = create_fallback_client()
        
        # Simple generation test
        prompt = "Hello, how are you today?"
        
        print(f"Testing generation with: '{prompt}'")
        
        result = client.generate_text(
            prompt,
            max_tokens=30,
            temperature=0.7
        )
        
        if result:
            print(f"🎉 SUCCESS! Generated: '{result}'")
            print("\n✅ The generation system works!")
            print("✅ Streamlit demo should work with OpenRouter option")
            
            # Test the actual streamlit generation function logic
            print("\n🧪 Testing Streamlit Integration")
            return test_streamlit_integration(client, prompt)
        else:
            print("❌ No generation result")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_streamlit_integration(client, test_prompt):
    """Test the streamlit generation integration."""
    try:
        # Replicate the openrouter generation from streamlit
        print("Testing OpenRouter generation like Streamlit demo...")
        
        generation_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        }
        
        result = client.generate_text(
            test_prompt,
            **generation_params
        )
        
        if result:
            print(f"✅ Streamlit-style generation works: '{result[:60]}...'")
            
            # Now test with different parameters
            creative_result = client.generate_text(
                "Write a creative story about robots:",
                max_tokens=80,
                temperature=0.9
            )
            
            if creative_result:
                print(f"✅ Creative generation works: '{creative_result[:60]}...'")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False

def main():
    """Main function to solve the generation issue."""
    print("🔧 Solving 'Doesn't Generate' Issue")
    print("=" * 50)
    
    # Step 1: Check environment
    env_ok = check_env_config()
    
    # Step 2: Test OpenRouter (should work immediately)
    openrouter_works = test_openrouter_generation()
    
    # Step 3: Test simple generation
    if env_ok and openrouter_works:
        generation_works = create_simple_generation_test()
        
        if generation_works:
            print("\n" + "=" * 50)
            print("🎉 ISSUE RESOLVED!")
            print("=" * 50)
            print("✅ OpenRouter generation is working")
            print("✅ Streamlit demo should work with 'OpenRouter Only' option")
            print("\n📋 How to use:")
            print("1. Open the Streamlit demo")
            print("2. Select 'OpenRouter Only' in the sidebar")
            print("3. Enter your prompt")
            print("4. Click 'Generate'")
            print("5. Generation should work immediately!")
            return True
    
    print("\n❌ Issue not fully resolved")
    print("   Check API key configuration in .env file")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)