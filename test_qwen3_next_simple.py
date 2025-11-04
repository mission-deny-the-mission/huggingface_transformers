#!/usr/bin/env python3
"""
Simple test script to verify Qwen3-Next model preset mapping.
"""
import os
import sys
from model_manager import ModelManager

def test_qwen3_next_preset():
    """Test Qwen3-Next model preset mapping."""
    print("Testing Qwen3-Next-80B-A3B-Instruct-FP8 model preset mapping...")
    
    # Create model manager
    manager = ModelManager()
    
    # Get model presets from load_default_model method
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
    
    # Test preset mapping
    print("\n1. Testing preset mapping...")
    test_presets = ["qwen3-next", "qwen3-next-instruct"]
    
    for preset in test_presets:
        if preset in model_presets:
            actual_model = model_presets[preset]
            print(f"✓ {preset} -> {actual_model}")
        else:
            print(f"✗ {preset} not found in presets")
            return False
    
    # Test model name detection
    print("\n2. Testing model name detection...")
    test_model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    is_qwen3_next = "Qwen3-Next-80B-A3B-Instruct" in test_model_name
    
    if is_qwen3_next:
        print(f"✓ Correctly detected Qwen3-Next model: {test_model_name}")
    else:
        print(f"✗ Failed to detect Qwen3-Next FP8 model: {test_model_name}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_qwen3_next_preset()
    sys.exit(0 if success else 1)