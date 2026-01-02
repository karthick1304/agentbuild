"""
🎓 LESSON 1: Basic LLM Call
============================
This demonstrates the simplest way to call an LLM - direct HTTP request.
No frameworks, no abstractions - just you and the API!
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def basic_llm_call(prompt: str, model: str = "qwen/qwen-2.5-72b-instruct") -> str:
    """
    The most fundamental LLM call - a simple HTTP POST request.
    
    This is what EVERY framework does under the hood!
    
    Args:
        prompt: Your question or instruction
        model: Which model to use (OpenRouter supports many!)
    
    Returns:
        The model's response text
    """
    
    # The request payload - this is the "language" LLMs speak
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Be concise."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,  # Controls randomness (0=deterministic, 1=creative)
        "max_tokens": 500    # Limit response length
    }
    
    # Headers for authentication
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Required by OpenRouter
    }
    
    # Make the actual HTTP request
    with httpx.Client() as client:
        response = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        
    # Parse the response
    result = response.json()
    return result["choices"][0]["message"]["content"]


def demonstrate_basic_call():
    """Run a demo of the basic LLM call"""
    print("=" * 60)
    print("🎓 LESSON 1: Basic LLM Call Demo")
    print("=" * 60)
    
    prompts = [
        "What is Python in one sentence?",
        "Write a haiku about coding",
        "What is 2 + 2? Just the number."
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 40)
        response = basic_llm_call(prompt)
        print(f"🤖 Response: {response}")
    
    print("\n" + "=" * 60)
    print("💡 KEY TAKEAWAY: An LLM call is just an HTTP request!")
    print("   - Send: model + messages + parameters")
    print("   - Receive: generated text")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_basic_call()

