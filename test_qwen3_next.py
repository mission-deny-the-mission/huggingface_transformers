#!/usr/bin/env python3
"""
Test script to verify Qwen3-Next model loading and basic functionality.
"""
import os
import sys
from model_manager import ModelManager

def test_qwen3_next():
    """Test loading and using the Qwen3-Next model."""
    print("Testing Qwen3-Next-80B-A3B-Instruct-FP8 model...")
    
    # Create model manager
    manager = ModelManager()
    
    # Test with preset name
    print("\n1. Testing with preset name 'qwen3-next'...")
    try:
        manager.load_model("qwen3-next")
        print("✓ Successfully loaded with preset name")
        manager.unload_model()
    except Exception as e:
        print(f"✗ Failed to load with preset name: {e}")
        return False
    
    # Test with full model name
    print("\n2. Testing with full model name...")
    try:
        manager.load_model("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
        print("✓ Successfully loaded with full model name")
        manager.unload_model()
    except Exception as e:
        print(f"✗ Failed to load with full model name: {e}")
        return False
    
    # Test message formatting
    print("\n3. Testing message formatting...")
    try:
        manager.load_model("qwen3-next")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        formatted = manager.format_messages(messages)
        print(f"✓ Message formatting successful")
        print(f"  Formatted prompt: {formatted[:100]}...")
        manager.unload_model()
    except Exception as e:
        print(f"✗ Failed to format messages: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_qwen3_next()
    sys.exit(0 if success else 1)