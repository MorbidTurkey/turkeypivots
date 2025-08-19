"""
Setup script for Turkey Pivots
Run this to set up the environment and install dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🦃 Turkey Pivots Setup Script")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"✗ Python 3.8+ required. You have {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create necessary directories
    directories = ['temp', 'data', 'backups', 'tests']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("\n❌ Failed to install dependencies. Please run manually:")
        print("pip install -r requirements.txt")
        return False
    
    # Test imports
    try:
        import dash
        import pandas as pd
        import plotly
        print("✓ Core dependencies imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Set up AI (choose one):")
    print("   - For Ollama (free): Install from https://ollama.ai and run 'ollama pull llama2'")
    print("   - For OpenAI: Create .env file with OPENAI_API_KEY=your_key")
    print("2. Run the application: python app.py")
    print("3. Open browser to: http://127.0.0.1:8050")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
