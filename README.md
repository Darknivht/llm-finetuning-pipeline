# 🚀 LLM Fine-Tuning Pipeline

A comprehensive, production-ready pipeline for fine-tuning Large Language Models with support for PEFT/LoRA, evaluation, and OpenRouter integration.

## ✨ Features

- **Multi-Method Training**: Full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT/LoRA)
- **Comprehensive Evaluation**: Automatic metrics computation with OpenRouter comparison
- **Data Processing**: Built-in cleaning, tokenization, and dataset preparation
- **Web Interface**: Streamlit demo for interactive testing
- **API Integration**: OpenRouter and OpenAI fallback support
- **Production Ready**: Docker support, logging, configuration management
- **Flexible**: Works with any HuggingFace model and custom datasets

## 🚦 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd llm-finetuning-pipeline

# Install dependencies
pip install -r requirements.txt

# Optional: Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Prepare Data

```bash
# Generate synthetic data for testing
python data_prep.py --generate_synthetic --output_dir ./data --synthetic_samples 1000

# Or prepare your own data
python data_prep.py --input_file your_data.csv --output_dir ./data --text_column text --label_column label
```

### 3. Train Model

```bash
# LoRA fine-tuning (recommended for beginners)
python train.py --data_path ./data/processed_dataset --output_dir ./checkpoints/lora_model --use_peft true

# Full fine-tuning
python train.py --data_path ./data/processed_dataset --output_dir ./checkpoints/full_model --use_peft false
```

### 4. Evaluate Model

```bash
python evaluate.py --checkpoint ./checkpoints/lora_model --dataset ./data/processed_dataset --output_dir ./eval_results
```

### 5. Run Inference

```bash
# Interactive mode
python inference.py --checkpoint ./checkpoints/lora_model --interactive

# Single prompt
python inference.py --checkpoint ./checkpoints/lora_model --input_text "The future of AI is"

# Web interface
streamlit run demo_streamlit.py
```

## 📁 Project Structure

```
llm-finetuning-pipeline/
├── backend/                 # Core backend modules
│   ├── __init__.py         # Package initialization and configuration
│   ├── models.py           # Model loading and PEFT setup
│   ├── data_utils.py       # Data processing and tokenization
│   ├── train_utils.py      # Training utilities and setup
│   ├── eval_utils.py       # Evaluation metrics and utilities
│   ├── openrouter.py       # OpenRouter API integration
│   └── logging_config.py   # Logging configuration
├── configs/                # Configuration files
│   ├── lora_small.yaml     # LoRA configuration
│   └── full_train.yaml     # Full training configuration
├── train.py                # Training script
├── evaluate.py            # Evaluation script
├── data_prep.py           # Data preparation script
├── inference.py           # Inference script
├── demo_streamlit.py      # Streamlit demo interface
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md            # This file
```

## 🔧 Detailed Usage

### Data Preparation

The pipeline supports various data formats and sources:

```bash
# From CSV/JSON file
python data_prep.py --input_file data.csv --output_dir ./data --text_column content --label_column category

# From HuggingFace dataset
python data_prep.py --hf_dataset imdb --output_dir ./data --text_column text --label_column label

# Generate synthetic data for testing
python data_prep.py --generate_synthetic --task_type completion --synthetic_samples 500 --output_dir ./data
```

### Training Configuration

You can use configuration files or command-line arguments:

```bash
# Using configuration file
python train.py --config configs/lora_small.yaml

# Using command-line arguments
python train.py \
    --model_name_or_path distilgpt2 \
    --data_path ./data/processed_dataset \
    --output_dir ./checkpoints/my_model \
    --epochs 3 \
    --batch_size 8 \
    --lr 2e-4 \
    --use_peft true \
    --lora_rank 16
```

### Advanced Training Options

```bash
# Resume from checkpoint
python train.py --config configs/lora_small.yaml --resume_from_checkpoint ./checkpoints/my_model/checkpoint-1000

# Mixed precision training
python train.py --data_path ./data --output_dir ./checkpoints --fp16 true

# Custom model
python train.py --model_name_or_path microsoft/DialoGPT-medium --data_path ./data --output_dir ./checkpoints
```

### Evaluation Options

```bash
# Basic evaluation
python evaluate.py --checkpoint ./checkpoints/my_model --dataset ./data/processed_dataset

# With OpenRouter comparison
python evaluate.py \
    --checkpoint ./checkpoints/my_model \
    --dataset ./data/processed_dataset \
    --compare_with_openrouter true \
    --openrouter_model gpt-3.5-turbo

# Custom evaluation parameters
python evaluate.py \
    --checkpoint ./checkpoints/my_model \
    --dataset ./data/processed_dataset \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --sample_size 100
```

### Inference Options

```bash
# Interactive mode
python inference.py --checkpoint ./checkpoints/my_model --interactive

# From file
python inference.py --checkpoint ./checkpoints/my_model --input_file prompts.txt --output_file results.json

# With OpenRouter comparison
python inference.py \
    --checkpoint ./checkpoints/my_model \
    --input_text "Complete this story: Once upon a time" \
    --compare_openrouter \
    --openrouter_model gpt-3.5-turbo
```

## 🌐 Streamlit Demo

Launch the interactive web interface:

```bash
streamlit run demo_streamlit.py
```

Features:
- Model loading and inference
- Parameter adjustment
- OpenRouter comparison
- Multiple input methods
- Real-time generation

## 🐳 Docker Usage

### Development Environment

```bash
# Build development image
docker build --target development -t llm-pipeline:dev .

# Run with volume mounting
docker run -it --gpus all -v $(pwd):/app -p 8501:8501 llm-pipeline:dev
```

### Production Deployment

```bash
# Build production image
docker build --target inference -t llm-pipeline:prod .

# Run inference server
docker run --gpus all -p 8501:8501 llm-pipeline:prod streamlit run demo_streamlit.py --server.port=8501 --server.address=0.0.0.0
```

### Training in Docker

```bash
# Build training image
docker build --target training -t llm-pipeline:train .

# Run training
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints llm-pipeline:train \
    python train.py --data_path /app/data --output_dir /app/checkpoints
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

# OpenRouter Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openrouter/auto

# Default paths
DATA_PATH=./data
OUTPUT_DIR=./checkpoints
```

### YAML Configuration

Example LoRA configuration (`configs/lora_small.yaml`):

```yaml
training:
  model_name_or_path: "distilgpt2"
  output_dir: "./checkpoints/lora_small"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2e-4
  use_peft: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1

data:
  data_path: "./data"
  text_column: "text"
  max_examples: 1000

eval:
  batch_size: 8
  max_new_tokens: 50
  temperature: 0.7
```

## 📊 Monitoring and Logging

The pipeline includes comprehensive logging:

- **Training logs**: Saved to `{output_dir}/logs/`
- **Evaluation results**: JSON and text summaries
- **Model checkpoints**: Automatic saving during training
- **Metrics tracking**: Loss, perplexity, and custom metrics

Example log structure:
```
logs/
├── training_2024-01-15_10-30-45.log
├── evaluation_2024-01-15_12-00-00.log
└── inference_2024-01-15_12-30-15.log
```

## 🎯 Use Cases

### 1. Text Completion
```bash
# Prepare data
python data_prep.py --generate_synthetic --task_type completion --output_dir ./data

# Train model
python train.py --config configs/lora_small.yaml --data_path ./data/processed_dataset

# Test
python inference.py --checkpoint ./checkpoints/lora_small --input_text "The benefits of AI include"
```

### 2. Text Classification
```bash
# Prepare classification data
python data_prep.py --input_file sentiment_data.csv --text_column review --label_column sentiment --output_dir ./data

# Train classifier
python train.py --data_path ./data/processed_dataset --task_type classification --output_dir ./checkpoints/classifier
```

### 3. Custom Domain Adaptation
```bash
# Use your domain-specific data
python data_prep.py --input_file medical_texts.jsonl --output_dir ./data/medical

# Fine-tune for medical domain
python train.py --model_name_or_path microsoft/BioGPT --data_path ./data/medical --output_dir ./checkpoints/medical-gpt
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python train.py --batch_size 4 --gradient_accumulation_steps 2
   
   # Use LoRA instead of full fine-tuning
   python train.py --use_peft true --lora_rank 8
   ```

2. **Model loading errors**
   ```bash
   # Check checkpoint structure
   ls -la ./checkpoints/your_model/
   
   # Verify configuration
   python -c "from backend.models import ModelLoader; ModelLoader.load_checkpoint('./checkpoints/your_model')"
   ```

3. **API connection issues**
   ```bash
   # Test OpenRouter connection
   python -c "from backend.openrouter import test_openrouter_connection; print(test_openrouter_connection())"
   ```

### Performance Optimization

1. **Memory optimization**
   - Use gradient checkpointing: `--gradient_checkpointing true`
   - Enable CPU offloading: `--dataloader_pin_memory false`
   - Reduce sequence length: `--max_length 256`

2. **Speed optimization**
   - Use mixed precision: `--fp16 true`
   - Increase batch size if memory allows
   - Use multiple GPUs: `--ddp true`

## 📈 Model Performance

### Evaluation Metrics

The pipeline automatically computes:
- **Perplexity**: Language modeling quality
- **BLEU Score**: Text generation quality
- **ROUGE Scores**: Summarization quality
- **Custom metrics**: Task-specific evaluations

### Benchmarking

Compare your model against baselines:
```bash
python evaluate.py \
    --checkpoint ./checkpoints/your_model \
    --dataset ./data/test_set \
    --compare_with_openrouter true \
    --openrouter_model gpt-3.5-turbo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the model framework
- [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning methods
- [OpenRouter](https://openrouter.ai/) for API integration
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Documentation**: See inline code documentation and this README

---

**Happy Fine-Tuning! 🚀**