#!/bin/bash

echo "Installing DCGAN for MNIST..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip first."
    exit 1
fi

echo "Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Packages installed successfully!"
else
    echo "✗ Error installing packages. Please check your internet connection and try again."
    exit 1
fi

echo "Running setup tests..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. For a quick test (5 epochs): python3 quick_test.py"
    echo "2. For full training (100 epochs): python3 train_dcgan.py"
    echo "3. To generate samples: python3 generate_samples.py --model_path checkpoints/final_model.pth"
    echo ""
    echo "Happy training! 🚀"
else
    echo ""
    echo "✗ Setup test failed. Please check the error messages above."
    exit 1
fi 

# Install Kaggle CLI for dataset download
pip install kaggle

# Instructions for user to set up Kaggle API key
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "\n[!] Kaggle API key not found. Please download your API key from https://www.kaggle.com > Account > Create New API Token."
  echo "   Place kaggle.json in ~/.kaggle/ and set permissions:"
  echo "   mkdir -p ~/.kaggle"
  echo "   mv /path/to/kaggle.json ~/.kaggle/"
  echo "   chmod 600 ~/.kaggle/kaggle.json"
fi 