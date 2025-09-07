#!/usr/bin/env python3
"""
Setup script for Document Summary Assistant
This script will install all required dependencies and download NLTK data
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk # type: ignore

        print("Downloading NLTK data...")
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
        print("The app will still work with basic text processing.")


def create_directories():
    """Create required directories"""
    directories = ["templates", "uploads"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")


def install_requirements():
    """Install all requirements"""
    requirements = [
        "Flask==2.3.3",
        "PyPDF2==3.0.1",
        "pytesseract==0.3.10",
        "Pillow==10.0.1",
        "nltk==3.8.1",
        "Werkzeug==2.3.7",
        "gunicorn==21.2.0",
        "python-dotenv==1.0.0",
    ]

    print("Installing Python packages...")
    for requirement in requirements:
        try:
            install_package(requirement)
            print(f"✅ Installed: {requirement}")
        except Exception as e:
            print(f"❌ Failed to install {requirement}: {e}")


def main():
    print("🚀 Setting up Document Summary Assistant...")
    print("=" * 50)

    # Install requirements
    install_requirements()
    print()

    # Create directories
    create_directories()
    print()

    # Download NLTK data
    download_nltk_data()
    print()

    print("=" * 50)
    print("🎉 Setup complete!")
    print()
    print("Next steps:")
    print("1. Make sure Tesseract OCR is installed on your system:")
    print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - macOS: brew install tesseract")
    print("   - Ubuntu/Debian: sudo apt install tesseract-ocr")
    print()
    print("2. Create templates/index.html with the HTML content")
    print("3. Run the application: python app.py")
    print("4. Open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
