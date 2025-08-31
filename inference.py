#!/usr/bin/env python3
"""
Inference script for the LLM Fine-Tuning Pipeline.
Supports local model inference with OpenRouter fallback comparison.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend import setup_logging
from backend.models import ModelLoader
from backend.openrouter import create_fallback_client
from backend.logging_config import get_logger

logger = get_logger("inference")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--base_model", type=str,
                       help="Base model path (for PEFT models)")
    
    # Input arguments
    parser.add_argument("--input_text", type=str,
                       help="Input text for generation")
    parser.add_argument("--input_file", type=str,
                       help="Path to file containing input text (one prompt per line)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--do_sample", type=bool, default=True,
                       help="Whether to use sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    
    # Output arguments
    parser.add_argument("--output_file", type=str,
                       help="Path to save generated outputs")
    parser.add_argument("--save_inputs", type=bool, default=True,
                       help="Save inputs along with outputs")
    
    # Comparison arguments
    parser.add_argument("--compare_openrouter", action="store_true",
                       help="Compare with OpenRouter model")
    parser.add_argument("--openrouter_model", type=str, default="openrouter/auto",
                       help="OpenRouter model for comparison")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def load_inputs_from_file(file_path: str) -> List[str]:
    """Load input texts from file."""
    inputs = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                inputs.append(line)
    
    logger.info(f"Loaded {len(inputs)} input prompts from {file_path}")
    return inputs


def generate_with_model(model, tokenizer, inputs: List[str], args) -> List[str]:
    """Generate text using the loaded model."""
    import torch
    from transformers import GenerationConfig
    
    logger.info(f"Generating text for {len(inputs)} prompts")
    
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample and args.temperature > 0,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    outputs = []
    device = next(model.parameters()).device
    
    for i, input_text in enumerate(inputs):
        if args.verbose:
            logger.info(f"Processing input {i+1}/{len(inputs)}")
        
        # Tokenize input
        inputs_tokenized = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs_tokenized,
                generation_config=generation_config
            )
        
        # Decode output (remove input tokens)
        input_length = inputs_tokenized.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        
        output_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        outputs.append(output_text.strip())
        
        if args.verbose:
            print(f"\nInput: {input_text}")
            print(f"Output: {output_text}")
            print("-" * 50)
    
    return outputs


def compare_with_openrouter(inputs: List[str], args) -> List[Optional[str]]:
    """Generate text using OpenRouter for comparison."""
    logger.info(f"Generating comparison text with OpenRouter model: {args.openrouter_model}")
    
    try:
        # Create fallback client (will use OpenRouter if available)
        fallback_client = create_fallback_client()
        
        outputs = []
        for i, input_text in enumerate(inputs):
            if args.verbose:
                logger.info(f"Processing OpenRouter input {i+1}/{len(inputs)}")
            
            output = fallback_client.generate_text(
                input_text,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                prefer_provider="openrouter"
            )
            outputs.append(output)
            
            if args.verbose and output:
                print(f"\nOpenRouter Output: {output}")
        
        return outputs
    
    except Exception as e:
        logger.error(f"OpenRouter comparison failed: {e}")
        return [None] * len(inputs)


def interactive_mode(model, tokenizer, args):
    """Run inference in interactive mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE INFERENCE MODE")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("-" * 60)
    
    # Initialize fallback client for comparison
    fallback_client = None
    if args.compare_openrouter:
        try:
            fallback_client = create_fallback_client()
            print("OpenRouter comparison enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter client: {e}")
            print("OpenRouter comparison disabled")
    
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("Enter text to generate from: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  quit/exit - Exit interactive mode")
                print("  Any other text - Generate completion")
                print()
                continue
            
            # Generate with local model
            print("\nGenerating...")
            local_output = generate_with_model(model, tokenizer, [user_input], args)[0]
            
            print(f"\nInput: {user_input}")
            print(f"Local Model: {local_output}")
            
            # Compare with OpenRouter if enabled
            if fallback_client:
                print("Generating OpenRouter comparison...")
                openrouter_output = fallback_client.generate_text(
                    user_input,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    prefer_provider="openrouter"
                )
                
                if openrouter_output:
                    print(f"OpenRouter: {openrouter_output}")
                else:
                    print("OpenRouter: Failed to generate")
            
            print("-" * 60)
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            print(f"Error: {e}")


def save_results(inputs: List[str], outputs: List[str], openrouter_outputs: Optional[List[str]], output_file: str):
    """Save generation results to file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, (input_text, output_text) in enumerate(zip(inputs, outputs)):
        result = {
            "input": input_text,
            "local_output": output_text,
            "local_output_length": len(output_text.split())
        }
        
        if openrouter_outputs and i < len(openrouter_outputs) and openrouter_outputs[i]:
            result["openrouter_output"] = openrouter_outputs[i]
            result["openrouter_output_length"] = len(openrouter_outputs[i].split())
        
        results.append(result)
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")
    
    # Also save as plain text for easy reading
    text_file = output_file.with_suffix('.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f"Example {i+1}:\n")
            f.write(f"Input: {result['input']}\n")
            f.write(f"Local Output: {result['local_output']}\n")
            
            if 'openrouter_output' in result:
                f.write(f"OpenRouter Output: {result['openrouter_output']}\n")
            
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Text results saved to {text_file}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger.info("Starting LLM inference")
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model, tokenizer, config_dict = ModelLoader.load_checkpoint(
        checkpoint_path, device=device
    )
    
    logger.info("Model loaded successfully")
    if config_dict and args.verbose:
        logger.info(f"Model configuration: {json.dumps(config_dict, indent=2, default=str)}")
    
    # Get inputs
    inputs = []
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
        return
    
    elif args.input_text:
        # Single input text
        inputs = [args.input_text]
    
    elif args.input_file:
        # Load from file
        inputs = load_inputs_from_file(args.input_file)
    
    else:
        # Read from stdin
        logger.info("Reading input from stdin (one prompt per line, Ctrl+D to finish):")
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    inputs.append(line)
        except KeyboardInterrupt:
            logger.info("Input interrupted by user")
    
    if not inputs:
        logger.error("No input provided")
        return
    
    logger.info(f"Processing {len(inputs)} input(s)")
    
    # Generate outputs
    try:
        outputs = generate_with_model(model, tokenizer, inputs, args)
        
        # Compare with OpenRouter if requested
        openrouter_outputs = None
        if args.compare_openrouter:
            openrouter_outputs = compare_with_openrouter(inputs, args)
        
        # Display results
        for i, (input_text, output_text) in enumerate(zip(inputs, outputs)):
            if not args.verbose and len(inputs) > 1:
                print(f"\nExample {i+1}:")
            
            if args.verbose or len(inputs) == 1:
                print(f"\nInput: {input_text}")
            
            print(f"Output: {output_text}")
            
            if openrouter_outputs and openrouter_outputs[i]:
                print(f"OpenRouter: {openrouter_outputs[i]}")
        
        # Save results if output file specified
        if args.output_file:
            save_results(inputs, outputs, openrouter_outputs, args.output_file)
        
        logger.info("Inference completed successfully!")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()