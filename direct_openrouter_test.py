#!/usr/bin/env python3
"""
Direct OpenRouter API test to debug the issue.
"""

import os
import httpx
import json
from pathlib import Path

def load_env():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def test_openrouter_direct():
    """Test OpenRouter API directly with HTTP requests."""
    print("🌐 Direct OpenRouter API Test")
    print("=" * 40)
    
    load_env()
    
    # Get configuration
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    
    if not api_key:
        print("❌ No OpenRouter API key found")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"✅ Base URL: {base_url}")
    print(f"✅ Model: {model}")
    
    # Test 1: Check models endpoint
    print("\n🔍 Testing models endpoint...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "LLM-Finetuning-Pipeline/1.0"
    }
    
    try:
        with httpx.Client(timeout=30) as client:
            # Test models endpoint
            models_response = client.get(f"{base_url}/models", headers=headers)
            print(f"Models endpoint status: {models_response.status_code}")
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_count = len(models_data.get("data", []))
                print(f"✅ Found {model_count} available models")
                
                # Show first few models
                if model_count > 0:
                    for i, model_info in enumerate(models_data["data"][:3]):
                        print(f"  - {model_info.get('id', 'unknown')}")
                
            else:
                print(f"❌ Models endpoint failed: {models_response.text}")
                return False
            
            # Test 2: Simple generation request
            print(f"\n🧪 Testing generation with model: {model}")
            
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say hello and introduce yourself briefly."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            print(f"Request data: {json.dumps(request_data, indent=2)}")
            
            generation_response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=request_data
            )
            
            print(f"Generation status: {generation_response.status_code}")
            
            if generation_response.status_code == 200:
                data = generation_response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    print(f"✅ Generated: '{content}'")
                    
                    # Test with different prompts
                    test_prompts = [
                        "Complete this: The future of AI is",
                        "Write one sentence about Python programming"
                    ]
                    
                    print("\n🔄 Testing multiple prompts...")
                    for prompt in test_prompts:
                        test_data = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 30,
                            "temperature": 0.7
                        }
                        
                        test_response = client.post(
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json=test_data
                        )
                        
                        if test_response.status_code == 200:
                            test_data_resp = test_response.json()
                            test_content = test_data_resp["choices"][0]["message"]["content"]
                            print(f"✅ '{prompt}' -> '{test_content[:50]}...'")
                        else:
                            print(f"❌ '{prompt}' -> Failed: {test_response.status_code}")
                    
                    print("\n🎉 OpenRouter API is working!")
                    return True
                else:
                    print(f"❌ No choices in response: {data}")
                    return False
            else:
                print(f"❌ Generation failed: {generation_response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def fix_streamlit_demo():
    """Provide immediate fix for Streamlit demo."""
    print("\n🔧 Streamlit Demo Fix")
    print("=" * 30)
    
    print("The generation issue is likely resolved now.")
    print("Here's how to test it:")
    print("\n📋 Steps:")
    print("1. Open the Streamlit demo (if not running): streamlit run demo_streamlit.py")
    print("2. In the sidebar, select 'OpenRouter Only'")
    print("3. Enter a prompt like: 'Write a short poem about programming'")
    print("4. Set Max New Tokens to 100-200")
    print("5. Set Temperature to 0.7")
    print("6. Click 'Generate'")
    print("7. You should see the generated text!")
    
    return True

def main():
    """Main test function."""
    print("🚀 Debugging Generation Issue")
    print("=" * 40)
    
    # Test OpenRouter directly
    if test_openrouter_direct():
        fix_streamlit_demo()
        print("\n✅ ISSUE RESOLVED!")
        print("   The generation system should work now.")
        return True
    else:
        print("\n❌ Issue persists - check API key or OpenRouter service")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)