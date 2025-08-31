#!/usr/bin/env python3
"""
Quick start script for the LLM Fine-Tuning Pipeline.
Provides an interactive setup and demo experience.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

def run_command(command, description="", check=True, shell=False):
    """Run a command with nice output formatting."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=check, capture_output=False)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=False)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with code {result.returncode}")
        
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def check_requirements():
    """Check if basic requirements are met."""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if pip is available
    try:
        import pip
        print("✅ pip is available")
    except ImportError:
        print("❌ pip not found")
        return False
    
    # Check for GPU (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available - will use CPU")
    except ImportError:
        print("⚠️ PyTorch not installed yet")
    
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install requirements
    success = run_command(
        "pip install -r requirements.txt",
        "Installing Python packages"
    )
    
    if success:
        print("✅ All dependencies installed successfully!")
    
    return success


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "data",
        "checkpoints", 
        "logs",
        "storage"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True


def generate_sample_data():
    """Generate sample data for demonstration."""
    print("\n📝 Generating sample data...")
    
    success = run_command(
        "python data_prep.py --generate_synthetic --synthetic_samples 100 --task_type completion --output_dir ./data/demo",
        "Generating synthetic demo data"
    )
    
    return success


def run_quick_training():
    """Run a quick training demonstration."""
    print("\n🚂 Running quick training demo...")
    
    success = run_command(
        "python train.py --data_path ./data/demo/processed_dataset --output_dir ./checkpoints/demo_model --epochs 1 --batch_size 4 --max_examples 50",
        "Training demo model (1 epoch, 50 samples)"
    )
    
    return success


def test_inference():
    """Test inference with the trained model."""
    print("\n🔮 Testing inference...")
    
    success = run_command(
        'python inference.py --checkpoint ./checkpoints/demo_model --input_text "The future of AI is"',
        "Testing model inference"
    )
    
    return success


def show_next_steps():
    """Show what users can do next."""
    print("\n🎉 Quick start completed!")
    print("=" * 60)
    print("Next steps:")
    print("\n1. 📊 Launch Streamlit demo:")
    print("   streamlit run demo_streamlit.py")
    print("\n2. 🔧 Train with your own data:")
    print("   python data_prep.py --input_file your_data.csv --output_dir ./data/your_data")
    print("   python train.py --data_path ./data/your_data --output_dir ./checkpoints/your_model")
    print("\n3. 📈 Evaluate your model:")
    print("   python evaluate.py --checkpoint ./checkpoints/your_model --dataset ./data/your_data")
    print("\n4. 🔍 Interactive inference:")
    print("   python inference.py --checkpoint ./checkpoints/demo_model --interactive")
    print("\n5. 📖 Read the full documentation in README.md")
    print("=" * 60)


def main():
    """Main quick start function."""
    print("🚀 LLM Fine-Tuning Pipeline - Quick Start")
    print("=" * 50)
    
    # Check if user wants to proceed
    response = input("\nThis will set up the LLM Fine-Tuning Pipeline and run a quick demo.\nProceed? (y/n): ")
    
    if response.lower() not in ['y', 'yes']:
        print("Quick start cancelled.")
        return
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please install Python 3.8+ and pip.")
        return
    
    # Step 2: Install dependencies
    print("\nInstalling dependencies (this may take a few minutes)...")
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your internet connection and try again.")
        return
    
    # Step 3: Set up directories
    setup_directories()
    
    # Step 4: Generate sample data
    if not generate_sample_data():
        print("\n❌ Failed to generate sample data.")
        return
    
    # Step 5: Ask if user wants to run training demo
    response = input("\nWould you like to run a quick training demo? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\nRunning training demo (this may take 2-5 minutes)...")
        
        if run_quick_training():
            # Step 6: Test inference
            print("\nTesting the trained model...")
            test_inference()
        else:
            print("\n⚠️ Training demo failed, but you can still use the pipeline.")
    else:
        print("\nSkipping training demo.")
    
    # Step 7: Show next steps
    show_next_steps()
    
    # Optional: Launch Streamlit demo
    response = input("\nWould you like to launch the Streamlit demo now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n🌐 Launching Streamlit demo...")
        print("The demo will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the demo server.")
        
        try:
            run_command(
                "streamlit run demo_streamlit.py",
                "Launching Streamlit demo",
                check=False
            )
        except KeyboardInterrupt:
            print("\nDemo server stopped.")
    
    print("\n✨ Quick start complete! Happy fine-tuning!")


if __name__ == "__main__":
    main()