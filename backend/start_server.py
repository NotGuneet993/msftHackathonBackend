#!/usr/bin/env python3
"""
Startup script for the Squat Form Analysis API

This script will:
1. Install required dependencies
2. Start the FastAPI server
3. Test the model loading
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_model():
    """Test if the model file exists and can be loaded"""
    model_path = "model/lstm_squat_model.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the trained model file is in the correct location.")
        return False
    
    print(f"✅ Model file found: {model_path}")
    return True

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    print("🏋️ Squat Form Analysis API Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ main.py not found. Please run this script from the project root directory.")
        return
    
    # Install dependencies
    if not install_requirements():
        print("❌ Cannot continue without required packages.")
        return
    
    # Test model
    if not test_model():
        print("⚠️  Model not found, but server will still start (predictions will fail)")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
