#!/usr/bin/env python3
"""
Evaluation and comparison example for the LLM Fine-Tuning Pipeline.
Shows how to evaluate models and compare with OpenRouter.
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend import setup_logging
from backend.models import ModelLoader
from backend.data_utils import generate_synthetic_data, DataProcessor
from backend.eval_utils import evaluate_model, TextGenerationEvaluator
from backend.openrouter import create_fallback_client


def create_evaluation_data():
    """Create evaluation dataset."""
    eval_prompts = [
        "The benefits of renewable energy include",
        "Machine learning algorithms can help",
        "The future of space exploration involves",
        "Climate change affects our planet by",
        "Artificial intelligence applications in healthcare",
        "The importance of cybersecurity in modern",
        "Sustainable development goals aim to",
        "Quantum computing might revolutionize",
        "The role of education in society",
        "Blockchain technology offers advantages"
    ]
    
    return eval_prompts


def evaluate_pretrained_model():
    """Evaluate a pretrained model as baseline."""
    print("\n📊 Evaluating Pretrained Model (Baseline)")
    print("-" * 50)
    
    # Load pretrained model
    model_name = "distilgpt2"
    model = ModelLoader.load_base_model(model_name, device="auto")
    tokenizer = ModelLoader.load_tokenizer(model_name)
    
    # Create evaluation data
    eval_prompts = create_evaluation_data()
    
    # Generate outputs
    evaluator = TextGenerationEvaluator(model, tokenizer)
    
    generated_texts = []
    for prompt in eval_prompts:
        output = evaluator.generate_text(
            prompt,
            max_new_tokens=30,
            temperature=0.7
        )
        generated_texts.append(output)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("-" * 30)
    
    # Compute perplexity
    test_texts = [prompt + " " + output for prompt, output in zip(eval_prompts, generated_texts)]
    perplexity = evaluator.compute_perplexity(test_texts)
    
    print(f"\n📈 Pretrained Model Results:")
    print(f"   Perplexity: {perplexity:.2f}")
    
    return {
        "model_type": "pretrained",
        "model_name": model_name,
        "prompts": eval_prompts,
        "outputs": generated_texts,
        "perplexity": perplexity
    }


def evaluate_finetuned_model():
    """Evaluate a fine-tuned model if available."""
    print("\n🎯 Evaluating Fine-tuned Model")
    print("-" * 50)
    
    # Look for available checkpoints
    checkpoints_dir = Path("./examples/outputs")
    available_models = []
    
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                final_dir = model_dir / "final"
                if final_dir.exists() and (final_dir / "adapter_config.json").exists():
                    available_models.append(final_dir)
    
    if not available_models:
        print("⚠️ No fine-tuned models found.")
        print("   Run basic_training.py or custom_data_training.py first.")
        return None
    
    # Use the first available model
    checkpoint_path = available_models[0]
    print(f"📁 Loading model from: {checkpoint_path}")
    
    try:
        model, tokenizer, config = ModelLoader.load_checkpoint(checkpoint_path, device="auto")
        print("✅ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create evaluation data
    eval_prompts = create_evaluation_data()
    
    # Generate outputs
    evaluator = TextGenerationEvaluator(model, tokenizer)
    
    generated_texts = []
    for prompt in eval_prompts:
        output = evaluator.generate_text(
            prompt,
            max_new_tokens=30,
            temperature=0.7
        )
        generated_texts.append(output)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        print("-" * 30)
    
    # Compute perplexity
    test_texts = [prompt + " " + output for prompt, output in zip(eval_prompts, generated_texts)]
    perplexity = evaluator.compute_perplexity(test_texts)
    
    print(f"\n📈 Fine-tuned Model Results:")
    print(f"   Model: {checkpoint_path}")
    print(f"   Perplexity: {perplexity:.2f}")
    if config:
        print(f"   Training samples: {config.get('training_samples', 'unknown')}")
        print(f"   LoRA rank: {config.get('lora_rank', 'N/A')}")
    
    return {
        "model_type": "finetuned",
        "checkpoint_path": str(checkpoint_path),
        "prompts": eval_prompts,
        "outputs": generated_texts,
        "perplexity": perplexity,
        "config": config
    }


def evaluate_openrouter_model():
    """Evaluate OpenRouter model if available."""
    print("\n🌐 Evaluating OpenRouter Model")
    print("-" * 50)
    
    try:
        fallback_client = create_fallback_client()
        
        # Test connection
        test_output = fallback_client.generate_text(
            "Test prompt",
            max_tokens=10,
            temperature=0.7,
            prefer_provider="openrouter"
        )
        
        if not test_output:
            print("⚠️ OpenRouter API not available or not configured")
            print("   Set OPENROUTER_BASE_URL and OPENROUTER_API_KEY to enable comparison")
            return None
        
        print("✅ OpenRouter API connection successful")
        
    except Exception as e:
        print(f"⚠️ OpenRouter not available: {e}")
        return None
    
    # Create evaluation data
    eval_prompts = create_evaluation_data()
    
    # Generate outputs
    generated_texts = []
    for i, prompt in enumerate(eval_prompts):
        print(f"Generating {i+1}/{len(eval_prompts)}: {prompt[:30]}...")
        
        output = fallback_client.generate_text(
            prompt,
            max_tokens=30,
            temperature=0.7,
            prefer_provider="openrouter"
        )
        
        if output:
            generated_texts.append(output)
            print(f"✅ Generated: {output}")
        else:
            generated_texts.append("[Generation failed]")
            print("❌ Generation failed")
    
    print(f"\n📈 OpenRouter Model Results:")
    print(f"   Successful generations: {sum(1 for text in generated_texts if text != '[Generation failed]')}/{len(generated_texts)}")
    
    return {
        "model_type": "openrouter",
        "prompts": eval_prompts,
        "outputs": generated_texts,
        "success_rate": sum(1 for text in generated_texts if text != "[Generation failed]") / len(generated_texts)
    }


def compare_results(results_list):
    """Compare results from different models."""
    print("\n🔍 Model Comparison")
    print("=" * 60)
    
    valid_results = [r for r in results_list if r is not None]
    
    if len(valid_results) < 2:
        print("⚠️ Need at least 2 models for comparison")
        return
    
    # Compare perplexity (if available)
    print("\n📊 Perplexity Comparison:")
    for result in valid_results:
        if "perplexity" in result:
            print(f"   {result['model_type'].title()}: {result['perplexity']:.2f}")
    
    # Compare generation quality (subjective)
    print("\n📝 Generation Quality Comparison:")
    print("   (First 3 examples)")
    
    for i in range(min(3, len(valid_results[0]["prompts"]))):
        prompt = valid_results[0]["prompts"][i]
        print(f"\n--- Prompt {i+1}: {prompt} ---")
        
        for result in valid_results:
            model_name = result["model_type"].title()
            if i < len(result["outputs"]):
                output = result["outputs"][i]
                print(f"{model_name:12}: {output}")
    
    # Summary metrics
    print(f"\n📈 Summary:")
    for result in valid_results:
        model_name = result["model_type"].title()
        print(f"\n{model_name}:")
        
        if "perplexity" in result:
            print(f"  Perplexity: {result['perplexity']:.2f}")
        
        if "success_rate" in result:
            print(f"  Success Rate: {result['success_rate']:.1%}")
        
        if "config" in result and result["config"]:
            config = result["config"]
            if "training_samples" in config:
                print(f"  Training Samples: {config['training_samples']}")


def save_comparison_results(results_list, output_file):
    """Save comparison results to file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        "evaluation_timestamp": str(Path(__file__).stat().st_mtime),
        "models_evaluated": len([r for r in results_list if r is not None]),
        "results": [r for r in results_list if r is not None]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"💾 Comparison results saved to: {output_file}")


def main():
    """Main evaluation and comparison function."""
    print("🔍 Model Evaluation and Comparison")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    results = []
    
    # Evaluate pretrained model (baseline)
    pretrained_result = evaluate_pretrained_model()
    results.append(pretrained_result)
    
    # Evaluate fine-tuned model (if available)
    finetuned_result = evaluate_finetuned_model()
    results.append(finetuned_result)
    
    # Evaluate OpenRouter model (if available)
    openrouter_result = evaluate_openrouter_model()
    results.append(openrouter_result)
    
    # Compare results
    compare_results(results)
    
    # Save results
    output_file = Path("./examples/outputs/evaluation_comparison.json")
    save_comparison_results(results, output_file)
    
    # Final summary
    print("\n🎉 Evaluation Comparison Completed!")
    print("=" * 50)
    
    valid_results = [r for r in results if r is not None]
    print(f"📊 Models evaluated: {len(valid_results)}")
    
    for result in valid_results:
        print(f"   ✅ {result['model_type'].title()}")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\nKey takeaways:")
    print("🔹 Lower perplexity generally indicates better language modeling")
    print("🔹 Fine-tuned models should show improved performance on target tasks")
    print("🔹 OpenRouter provides strong baseline comparisons")
    print("🔹 Quality depends on training data and hyperparameters")
    
    print("\nNext steps:")
    print("🔹 Analyze the generated outputs for quality and relevance")
    print("🔹 Try different hyperparameters if results are not satisfactory")
    print("🔹 Collect more training data for better fine-tuning results")
    print("🔹 Experiment with different base models")


if __name__ == "__main__":
    main()