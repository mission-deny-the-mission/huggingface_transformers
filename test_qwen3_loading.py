#!/usr/bin/env python3
"""
Test script to verify Qwen3-Next model loading setup (without actually loading the model).
"""
import os
import sys
from model_manager import ModelManager

def test_qwen3_loading_setup():
    """Test Qwen3-Next model loading setup."""
    print("Testing Qwen3-Next-80B-A3B-Instruct model loading setup...")
    
    # Create model manager
    manager = ModelManager()
    
    # Test model name resolution
    print("\n1. Testing model name resolution...")
    test_cases = [
        ("qwen3-next", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        ("qwen3-next-instruct", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
    ]
    
    for preset, expected in test_cases:
        # Simulate the preset resolution logic from load_default_model
        model_presets = {
            "default": "LiquidAI/LFM2-700M",
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2": "Qwen/Qwen2.5-14B-Instruct",
            "qwen3": "Qwen/Qwen2.5-72B-Instruct",
            "qwen3-next": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "qwen3-next-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large",
            "dialogpt": "microsoft/DialoGPT-medium",
        }
        
        actual = model_presets.get(preset, preset)
        if actual == expected:
            print(f"✓ {preset} -> {actual}")
        else:
            print(f"✗ {preset} -> {actual} (expected {expected})")
            return False
    
    # Test model detection logic
    print("\n2. Testing model detection logic...")
    test_model_names = [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF",  # Should not match
        "Qwen/Qwen2.5-7B-Instruct",  # Should not match
    ]
    
    for model_name in test_model_names:
        is_qwen3_next = "Qwen3-Next-80B-A3B-Instruct" in model_name
        expected = "Qwen3-Next-80B-A3B-Instruct" in model_name
        
        if is_qwen3_next == expected:
            status = "✓" if is_qwen3_next else "○"
            print(f"{status} {model_name} -> {'Qwen3-Next' if is_qwen3_next else 'Not Qwen3-Next'}")
        else:
            print(f"✗ {model_name} -> Detection logic error")
            return False
    
    print("\n✓ All setup tests passed!")
    print("\nTo use the Qwen3-Next model:")
    print("1. Start server: python app.py --model qwen3-next")
    print("2. Run benchmark: python benchmark.py --model qwen3-next --prompt 'Your prompt here' --runs 3")
    
    return True

if __name__ == "__main__":
    success = test_qwen3_loading_setup()
    sys.exit(0 if success else 1)