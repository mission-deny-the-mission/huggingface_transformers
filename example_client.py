"""
Example client for testing the OpenAI-compatible API.
"""
import requests
import json


def test_chat_completion():
    """Test the chat completion endpoint."""
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about artificial intelligence."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("Sending request to API...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("\nResponse:")
        print(json.dumps(result, indent=2))
        
        if result.get("choices"):
            print("\nGenerated text:")
            print(result["choices"][0]["message"]["content"])
            
        if result.get("usage"):
            usage = result["usage"]
            print(f"\nTokens used: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on http://localhost:8000")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    test_chat_completion()

