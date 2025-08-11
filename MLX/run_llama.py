#!/usr/bin/env python3
"""
MLX LLaMA-2-13B Runner
Runs LLaMA-2-13B in 4-bit quantization using Apple's MLX framework.
"""

import argparse
import time
import sys
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.utils import generate_step
except ImportError as e:
    print(f"Error: Missing required dependencies")
    print(f"Import error: {e}")
    sys.exit(1)

def load_model(model_path: str = None):
    """Load the LLaMA-2-13B model with 4-bit quantization."""
    if model_path is None:
        # Default to a 4-bit quantized LLaMA-2-13B model
        model_path = "TheBloke/Llama-2-13B-chat-AWQ"
    
    print(f"Loading model: {model_path}")
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load(model_path)
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sufficient memory (32GB+ recommended)")
        sys.exit(1)

def generate_text(model, tokenizer, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    """Generate text using the loaded model."""
    print(f"\nGenerating response...")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Tokenize the prompt
        inputs = tokenizer.encode(prompt)
        
        # Generate response
        response = generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        generation_time = time.time() - start_time
        tokens_generated = len(tokenizer.encode(response)) - len(inputs)
        
        print(f"Response:\n{response}")
        print("-" * 50)
        print(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
        print(f"Speed: {tokens_generated/generation_time:.1f} tokens/second")
        
        return response
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run LLaMA-2-13B with MLX")
    parser.add_argument("--prompt", type=str, default="Explain what machine learning is in simple terms", 
                       help="Input prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Sampling temperature (0.0 = deterministic, 1.0 = random)")
    parser.add_argument("--model", type=str, 
                       help="Custom model path (default: TheBloke/Llama-2-13B-chat-AWQ)")
    
    args = parser.parse_args()
    
    print("MLX LLaMA-2-13B Runner")
    print("=" * 50)
    
    # Check MLX version and device info
    print(f"MLX version: {mx.__version__}")
    print(f"Available devices: {mx.metal.get_device_count()}")
    if mx.metal.is_available():
        print(f"Metal device: {mx.metal.get_device_name()}")
    print(f"Default device: {mx.default_device()}")
    print()
    
    # Load the model
    model, tokenizer = load_model(args.model)
    
    # Generate text
    response = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        args.max_tokens, 
        args.temperature
    )
    
    if response:
        print("\nGeneration completed successfully!")
    else:
        print("\nGeneration failed!")

if __name__ == "__main__":
    main() 