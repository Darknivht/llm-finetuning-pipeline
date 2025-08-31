# Multi-stage Dockerfile for LLM Fine-Tuning Pipeline
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p logs storage data checkpoints

# Expose ports
EXPOSE 8501

# Create non-root user
RUN useradd -m -u 1000 llmuser && chown -R llmuser:llmuser /app
USER llmuser

# Default command (can be overridden)
CMD ["python", "-m", "streamlit", "run", "demo_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage
FROM base as development

USER root
RUN pip install --no-cache-dir jupyter ipython pytest

# Install additional development tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    isort \
    mypy

USER llmuser

# Training stage (for dedicated training containers)
FROM base as training

# Set default command for training
CMD ["python", "train.py", "--help"]

# Inference stage (optimized for serving)
FROM base as inference

# Install only minimal dependencies for inference
USER root
RUN pip uninstall -y \
    datasets \
    transformers[sentencepiece] || true

# Reinstall core inference dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    transformers==4.35.0 \
    tokenizers \
    peft==0.3.0

USER llmuser

# Set default command for inference
CMD ["python", "inference.py", "--help"]