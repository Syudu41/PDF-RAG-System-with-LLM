#!/usr/bin/env python3
"""
Setup script for PDF RAG system
Run this first to install dependencies and test basic functionality
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\nTesting imports...")
    
    required_packages = [
        ("fitz", "PyMuPDF"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("streamlit", "streamlit"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("requests", "requests")
    ]
    
    failed_imports = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (install with: pip install {pip_name})")
            failed_imports.append(pip_name)
    
    if failed_imports:
        print(f"\nMissing packages: {', '.join(failed_imports)}")
        return False
    else:
        print("✓ All imports successful!")
        return True

def create_directories():
    """Create necessary directories"""
    dirs = ["./data", "./chroma_db", "./temp"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def main():
    print("PDF RAG System Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Setup failed at requirements installation.")
        return False
    
    # Test imports
    print("\n3. Testing imports...")
    if not test_imports():
        print("Setup failed at import testing.")
        return False
    
    print("\n" + "=" * 40)
    print("✓ Setup completed successfully!")
    print("✓ You can now run: python test_pdf_processor.py")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    main()