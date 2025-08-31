#!/usr/bin/env python3
"""Setup script for the LLM Fine-Tuning Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").strip().split("\n")
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="llm-finetuning-pipeline",
    version="1.0.0",
    author="LLM Pipeline Team",
    author_email="team@llmpipeline.com",
    description="A comprehensive pipeline for fine-tuning Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Darknivht/llm-finetuning-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "llm-train=train:main",
            "llm-eval=evaluate:main",
            "llm-prep=data_prep:main",
            "llm-inference=inference:main",
            "llm-demo=demo_streamlit:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Darknivhg/llm-finetuning-pipeline/issues",
        "Documentation": "https://github.com/Darknivht/llm-finetuning-pipeline/blob/main/README.md",
        "Source": "https://github.com/Darknivht/llm-finetuning-pipeline",
    },
    keywords=[
        "machine learning",
        "deep learning",
        "transformers",
        "fine-tuning",
        "llm",
        "nlp",
        "peft",
        "lora",
        "pytorch",
        "huggingface",
    ],
)