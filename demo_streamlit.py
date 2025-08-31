#!/usr/bin/env python3
"""
Streamlit demo for the LLM Fine-Tuning Pipeline.
Provides a web interface for testing model inference and OpenRouter comparison.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.models import ModelLoader
from backend.openrouter import create_fallback_client
from backend.logging_config import setup_logging

# Configure Streamlit page
st.set_page_config(
    page_title="LLM Fine-Tuning Pipeline Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging(log_level="INFO", enable_console=False)

# Page header
st.title("🚀 LLM Fine-Tuning Pipeline Demo")
st.markdown("Test your fine-tuned models with an interactive web interface")


@st.cache_resource
def load_model_cached(checkpoint_path: str, device: str = "auto"):
    """Load model with caching to avoid reloading."""
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    try:
        with st.spinner("Loading model..."):
            model, tokenizer, config = ModelLoader.load_checkpoint(checkpoint_path, device)
        st.success("Model loaded successfully!")
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None


@st.cache_resource
def create_fallback_client_cached():
    """Create fallback client with caching."""
    try:
        return create_fallback_client()
    except Exception as e:
        st.warning(f"Failed to initialize fallback client: {e}")
        return None


def generate_text_local(model, tokenizer, prompt: str, **kwargs) -> Optional[str]:
    """Generate text using local model."""
    import torch
    from transformers import GenerationConfig
    
    try:
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_new_tokens", 100),
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
        st.error(f"Local generation failed: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    
    # Find available checkpoints
    checkpoints_dir = Path("./checkpoints")
    available_checkpoints = []
    
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid checkpoint
                if (item / "config.json").exists() or (item / "adapter_config.json").exists():
                    available_checkpoints.append(str(item))
    
    use_local_model = st.sidebar.radio(
        "Choose inference method:",
        ["Local Model", "OpenRouter Only"],
        help="Select whether to use a local fine-tuned model or OpenRouter API"
    )
    
    model, tokenizer, config = None, None, None
    
    if use_local_model == "Local Model":
        if available_checkpoints:
            checkpoint_path = st.sidebar.selectbox(
                "Select checkpoint:",
                available_checkpoints,
                help="Choose a model checkpoint to load"
            )
            
            if st.sidebar.button("Load Model"):
                model, tokenizer, config = load_model_cached(checkpoint_path)
        else:
            st.sidebar.warning("No checkpoints found in ./checkpoints/")
            st.sidebar.info("Train a model first or specify a checkpoint directory")
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    max_new_tokens = st.sidebar.slider(
        "Max New Tokens",
        min_value=10,
        max_value=500,
        value=100,
        help="Maximum number of new tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Sampling temperature (0.0 = deterministic)"
    )
    
    top_k = st.sidebar.slider(
        "Top K",
        min_value=1,
        max_value=100,
        value=50,
        help="Top-k sampling parameter"
    )
    
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Top-p (nucleus) sampling parameter"
    )
    
    # OpenRouter configuration
    st.sidebar.subheader("OpenRouter Settings")
    
    enable_openrouter = st.sidebar.checkbox(
        "Enable OpenRouter Comparison",
        value=False,
        help="Compare results with OpenRouter model"
    )
    
    openrouter_model = st.sidebar.text_input(
        "OpenRouter Model",
        value="openrouter/auto",
        help="OpenRouter model to use for comparison"
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        
        # Input methods
        input_method = st.radio(
            "Input method:",
            ["Text Input", "File Upload", "Example Prompts"]
        )
        
        input_text = ""
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter your prompt:",
                height=150,
                placeholder="Type your text here..."
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt'],
                help="Upload a text file with your prompt"
            )
            
            if uploaded_file is not None:
                input_text = uploaded_file.read().decode("utf-8")
                st.text_area("File content:", value=input_text, height=150, disabled=True)
        
        elif input_method == "Example Prompts":
            examples = [
                "The weather today is",
                "In the future, artificial intelligence will",
                "The best way to learn programming is",
                "My favorite book is about",
                "The most important skill in the 21st century is"
            ]
            
            selected_example = st.selectbox("Choose an example:", [""] + examples)
            if selected_example:
                input_text = selected_example
                st.text_area("Selected prompt:", value=input_text, disabled=True)
    
    with col2:
        st.header("Output")
        
        if st.button("Generate", type="primary", disabled=not input_text.strip()):
            if not input_text.strip():
                st.warning("Please enter some text to generate from")
            else:
                generation_params = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p
                }
                
                # Generate with local model
                local_output = None
                if use_local_model == "Local Model" and model and tokenizer:
                    with st.spinner("Generating with local model..."):
                        start_time = time.time()
                        local_output = generate_text_local(
                            model, tokenizer, input_text, **generation_params
                        )
                        local_time = time.time() - start_time
                    
                    if local_output:
                        st.subheader("Local Model Output")
                        st.write(local_output)
                        st.caption(f"Generated in {local_time:.2f} seconds")
                    else:
                        st.error("Local model generation failed")
                
                # Generate with OpenRouter if enabled
                openrouter_output = None
                if enable_openrouter:
                    fallback_client = create_fallback_client_cached()
                    
                    if fallback_client:
                        with st.spinner("Generating with OpenRouter..."):
                            start_time = time.time()
                            openrouter_output = fallback_client.generate_text(
                                input_text,
                                max_tokens=max_new_tokens,
                                temperature=temperature,
                                prefer_provider="openrouter"
                            )
                            openrouter_time = time.time() - start_time
                        
                        if openrouter_output:
                            st.subheader("OpenRouter Output")
                            st.write(openrouter_output)
                            st.caption(f"Generated in {openrouter_time:.2f} seconds")
                        else:
                            st.error("OpenRouter generation failed")
                    else:
                        st.error("OpenRouter client not available")
                
                # Show comparison if both outputs exist
                if local_output and openrouter_output:
                    st.subheader("Comparison")
                    
                    # Simple metrics
                    local_words = len(local_output.split())
                    openrouter_words = len(openrouter_output.split())
                    
                    comparison_data = {
                        "Metric": ["Word Count", "Character Count"],
                        "Local Model": [local_words, len(local_output)],
                        "OpenRouter": [openrouter_words, len(openrouter_output)]
                    }
                    
                    st.table(comparison_data)
    
    # Additional information sections
    st.markdown("---")
    
    # Model information
    if config and use_local_model == "Local Model":
        with st.expander("Model Information"):
            st.json(config)
    
    # Usage instructions
    with st.expander("Usage Instructions"):
        st.markdown("""
        ### How to use this demo:
        
        1. **Model Selection**: Choose between using a local fine-tuned model or OpenRouter API
        2. **Input**: Enter your prompt using text input, file upload, or example prompts
        3. **Parameters**: Adjust generation parameters in the sidebar
        4. **Generate**: Click the Generate button to create text
        5. **Compare**: Enable OpenRouter comparison to see different model outputs
        
        ### Tips:
        - Lower temperature (0.1-0.3) for more focused, deterministic outputs
        - Higher temperature (0.7-1.0) for more creative, diverse outputs
        - Use Top-K and Top-P to control output diversity
        - Enable OpenRouter comparison to evaluate your model's performance
        """)
    
    # Environment information
    with st.expander("Environment Information"):
        env_info = {
            "Available Models": len(available_checkpoints),
            "OpenRouter Available": bool(os.getenv("OPENROUTER_BASE_URL")),
            "OpenAI Available": bool(os.getenv("OPENAI_API_KEY"))
        }
        
        for key, value in env_info.items():
            st.write(f"**{key}**: {value}")
        
        if not any([os.getenv("OPENROUTER_BASE_URL"), os.getenv("OPENAI_API_KEY")]):
            st.info("Set OPENROUTER_BASE_URL or OPENAI_API_KEY environment variables to enable API comparisons")


if __name__ == "__main__":
    main()