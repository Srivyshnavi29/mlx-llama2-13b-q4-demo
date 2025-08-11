#!/bin/bash

# MLX LLaMA-2-13B Setup Script
# Automates the setup process for running LLaMA-2-13B with MLX

set -e

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is designed for macOS with Apple Silicon"
    exit 1
fi

# Check if we have Apple Silicon
if ! sysctl -n machdep.cpu.brand_string | grep -q "Apple"; then
    echo "This script requires Apple Silicon (M1/M2/M3)"
    exit 1
fi

echo "Detected Apple Silicon Mac"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYTHON_VERSION detected"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is required but not installed"
    echo "Please install pip3 or upgrade Python"
    exit 1
fi

echo "pip3 detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Dependencies installed successfully!"

# Check MLX installation
echo "Verifying MLX installation..."
python3 -c "
import mlx.core as mx
print(f'MLX {mx.__version__} installed successfully')
print(f'Metal available: {mx.metal.is_available()}')
if mx.metal.is_available():
    print(f'Metal device: {mx.metal.get_device_name()}')
"

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_llama.py
chmod +x chat_llama.py
chmod +x benchmark.py

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run a quick test: python run_llama.py"
echo "3. Start interactive chat: python chat_llama.py"
echo "4. Run benchmarks: python benchmark.py"
echo ""
echo "Note: The first run will download the LLaMA-2-13B model (~7.5GB)"
echo "This may take several minutes depending on your internet connection."
echo ""