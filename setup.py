"""Setup script for installing dependencies and initializing the system."""

import subprocess
import sys
from pathlib import Path
import shutil


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print("✓ Python version is compatible")


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install dependencies: {e}")
        sys.exit(1)


def setup_environment():
    """Set up environment file."""
    print_header("Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists, skipping...")
    else:
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("✓ Created .env file from template")
            print("\n⚠️  IMPORTANT: Edit .env and add your DeepSeek API key!")
        else:
            print("❌ .env.example not found")


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")
    
    directories = [
        "data",
        "data/vector_db",
        "data/feedback",
        "logs"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}/")


def verify_installation():
    """Verify installation."""
    print_header("Verifying Installation")
    
    try:
        # Try importing key modules
        import chromadb
        import sentence_transformers
        import streamlit
        from openai import OpenAI
        
        print("✓ ChromaDB installed")
        print("✓ Sentence Transformers installed")
        print("✓ Streamlit installed")
        print("✓ OpenAI client installed")
        
        print("\n✅ All core dependencies verified!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Try running: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main setup function."""
    print("\n" + "🧠 Self-Improving RAG System - Setup".center(80))
    
    check_python_version()
    install_dependencies()
    setup_environment()
    create_directories()
    verify_installation()
    
    print_header("Setup Complete!")
    
    print("Next steps:")
    print("1. Edit .env and add your DeepSeek API key")
    print("2. Run examples: python examples.py")
    print("3. Launch dashboard: python main.py dashboard")
    print("4. Or use CLI: python main.py --help")
    
    print("\n" + "="*80)
    print("Happy building! 🚀")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
