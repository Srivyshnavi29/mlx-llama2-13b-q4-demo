#!/usr/bin/env python3
"""
MLX LLaMA-2-13B Benchmark Script
Tests performance, memory usage, and generation speed.
"""

import time
import psutil
import argparse
import sys
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError as e:
    print(f"Error: Missing required dependencies")
    print(f"Import error: {e}")
    sys.exit(1)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024**3)  # Convert to GB

def benchmark_model(model_path: str = None, num_runs: int = 5):
    """Run benchmark tests on the model."""
    print("MLX LLaMA-2-13B Benchmark")
    print("=" * 50)
    
    # Test prompts of varying lengths
    test_prompts = [
        "Hello, how are you?",
        "Explain the concept of machine learning in detail.",
        "Write a short story about a robot learning to paint.",
        "Describe the process of photosynthesis step by step with scientific accuracy.",
        "Generate a comprehensive analysis of the impact of artificial intelligence on modern society, including economic, social, and ethical considerations."
    ]
    
    print(f"Loading model: {model_path or 'TheBloke/Llama-2-13B-chat-AWQ'}")
    print("This may take a few minutes on first run...")
    
    # Load model and measure memory
    start_memory = get_memory_usage()
    load_start = time.time()
    
    model, tokenizer = load(model_path or "TheBloke/Llama-2-13B-chat-AWQ")
    
    load_time = time.time() - load_start
    after_load_memory = get_memory_usage()
    
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Memory usage: {start_memory:.2f}GB → {after_load_memory:.2f}GB (+{after_load_memory - start_memory:.2f}GB)")
    print()
    
    # Benchmark generation for each prompt
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"   Input tokens: {prompt_tokens}")
        
        # Run multiple times for averaging
        generation_times = []
        token_counts = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9
            )
            
            generation_time = time.time() - start_time
            response_tokens = len(tokenizer.encode(response))
            
            generation_times.append(generation_time)
            token_counts.append(response_tokens)
            
            if run == 0:  # Show first response as sample
                print(f"   Sample response: {response[:100]}{'...' if len(response) > 100 else ''}")
        

        avg_generation_time = sum(generation_times) / len(generation_times)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_tokens_per_second = avg_tokens / avg_generation_time
        
        results.append({
            'prompt': prompt,
            'input_tokens': prompt_tokens,
            'avg_output_tokens': avg_tokens,
            'avg_generation_time': avg_generation_time,
            'avg_tokens_per_second': avg_tokens_per_second
        })
        
        print(f" Avg generation time: {avg_generation_time:.2f}s")
        print(f" Avg output tokens: {avg_tokens:.1f}")
        print(f" Avg speed: {avg_tokens_per_second:.1f} tokens/second")
        print()
    
    # Summary
    print("BENCHMARK SUMMARY")
    
    total_input_tokens = sum(r['input_tokens'] for r in results)
    total_output_tokens = sum(r['avg_output_tokens'] for r in results)
    total_time = sum(r['avg_generation_time'] for r in results)
    overall_speed = total_output_tokens / total_time
    
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens:.1f}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Overall generation speed: {overall_speed:.1f} tokens/second")
    print(f"Model memory footprint: {after_load_memory - start_memory:.2f}GB")
    
    if overall_speed >= 15:
        rating = "Excellent"
    elif overall_speed >= 10:
        rating = "Good"
    elif overall_speed >= 5:
        rating = "Acceptable"
    else:
        rating = "Slow"
    
    print(f"Performance rating: {rating}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLaMA-2-13B with MLX")
    parser.add_argument("--model", type=str, 
                       help="Custom model path (default: TheBloke/Llama-2-13B-chat-AWQ)")
    parser.add_argument("--runs", type=int, default=5, 
                       help="Number of runs per test for averaging")
    
    args = parser.parse_args()
    
    try:
        results = benchmark_model(args.model, args.runs)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("MLX LLaMA-2-13B Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Input tokens: {result['input_tokens']}\n")
                f.write(f"Output tokens: {result['avg_output_tokens']:.1f}\n")
                f.write(f"Generation time: {result['avg_generation_time']:.2f}s\n")
                f.write(f"Speed: {result['avg_tokens_per_second']:.1f} tokens/second\n\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n Benchmark failed: {e}")

if __name__ == "__main__":
    main() 