#!/usr/bin/env python3
"""
Interactive Chat with LLaMA-2-13B using MLX
Provides a conversational interface to the model.
"""

import argparse
import time
import sys
import os

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install with: pip install -r requirements.txt")
    print(f"Import error: {e}")
    sys.exit(1)

class LLaMAChat:
    def __init__(self, model_path: str = None, max_tokens: int = 512, temperature: float = 0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []
        
        print("Loading LLaMA-2-13B model...")
        self.model, self.tokenizer = load(model_path or "TheBloke/Llama-2-13B-chat-AWQ")
        print("Model loaded! Ready to chat.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'help' for available commands.")
        print("-" * 60)
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        print("Conversation history cleared.")
    
    def format_prompt(self, user_input: str) -> str:
        """Format the user input with conversation history for better context."""
        if not self.conversation_history:
            return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        # Build context from recent history (last 4 exchanges to avoid token limit)
        recent_history = self.conversation_history[-8:]  # Last 4 exchanges
        
        formatted_prompt = ""
        for msg in recent_history:
            if msg["role"] == "user":
                formatted_prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            else:
                formatted_prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
        formatted_prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        return formatted_prompt
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response from the model."""
        try:
            formatted_prompt = self.format_prompt(user_input)
            
            start_time = time.time()
            response = generate(
                self.model,
                self.tokenizer,
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            
            # Clean up the response
            response = response.strip()
            if response.startswith("assistant\n"):
                response = response[10:].strip()
            
            generation_time = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(response))
            
            print(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def chat(self):
        """Main chat loop."""
        print("Start chatting with LLaMA-2-13B!")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                # Add user input to history
                self.add_to_history("user", user_input)
                
                # Generate and display response
                print("\nLLaMA-2-13B: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Add response to history
                self.add_to_history("assistant", response)
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\nEnd of input. Goodbye!")
                break
    
    def show_help(self):
        """Show available commands."""
        print("\nAvailable Commands:")
        print("  help     - Show this help message")
        print("  clear    - Clear conversation history")
        print("  history  - Show conversation history")
        print("  stats    - Show conversation statistics")
        print("  quit     - Exit the chat")
        print("  exit     - Exit the chat")
        print("  bye      - Exit the chat")
    
    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print(f"\nConversation History ({len(self.conversation_history)} messages):")
        for i, msg in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    
    def show_stats(self):
        """Show conversation statistics."""
        total_messages = len(self.conversation_history)
        user_messages = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.conversation_history if msg["role"] == "assistant")
        
        print(f"\nConversation Statistics:")
        print(f"  Total messages: {total_messages}")
        print(f"  User messages: {user_messages}")
        print(f"  Assistant messages: {assistant_messages}")
        print(f"  Model: LLaMA-2-13B (4-bit quantized)")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Temperature: {self.temperature}")

def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with LLaMA-2-13B")
    parser.add_argument("--model", type=str, 
                       help="Custom model path (default: TheBloke/Llama-2-13B-chat-AWQ)")
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Sampling temperature (0.0 = deterministic, 1.0 = random)")
    
    args = parser.parse_args()
    
            print("MLX LLaMA-2-13B Interactive Chat")
    print("=" * 60)
    
    # Check MLX version
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print()
    
    # Start the chat
    chat = LLaMAChat(args.model, args.max_tokens, args.temperature)
    chat.chat()

if __name__ == "__main__":
    main() 