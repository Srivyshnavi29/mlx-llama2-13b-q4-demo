# MLX LLaMA-2-13B Project

This project demonstrates running LLaMA-2-13B in 4-bit quantization using Apple's MLX framework on Apple Silicon (M2 Max with 64GB unified memory).

## Hardware Requirements

- **Apple Silicon Mac** (M1/M2/M3 series)
- **Minimum 32GB unified memory** (64GB recommended for optimal performance)
- **macOS 13.0+**

## Features

- **4-bit quantization** for memory efficiency
- **Fully in-memory** operation (no disk swapping)
- **Native Apple Silicon optimization**
- **Fast inference** with MLX acceleration

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the model** (first run will download automatically):
   ```bash
   python run_llama.py
   ```

## Usage

### Basic Inference
```bash
python run_llama.py
```

### Interactive Chat Mode
```bash
python chat_llama.py
```

### Custom Prompts
```bash
python run_llama.py --prompt "Explain quantum computing in simple terms"
```

## Model Details

- **Model**: LLaMA-2-13B
- **Quantization**: 4-bit (GPTQ/AWQ compatible)
- **Memory Usage**: ~8-12GB (4-bit quantized)
- **Context Length**: 4096 tokens
- **Performance**: Optimized for Apple Silicon

## Performance Notes

- **First run**: Model download (~7.5GB for 4-bit quantized)
- **Subsequent runs**: Instant loading from memory
- **Inference speed**: ~10-20 tokens/second on M2 Max
- **Memory efficiency**: ~75% reduction vs 16-bit

## Troubleshooting

- **Out of memory**: Ensure you have at least 32GB unified memory
- **Slow performance**: Close other memory-intensive applications
- **Model download issues**: Check internet connection and HuggingFace access