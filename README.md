# OpenAI-Compatible Wrapper for Hugging Face Transformers

A production-ready OpenAI-compatible API wrapper for Hugging Face Transformers models, with built-in benchmarking tools.

## Features

- ✅ **OpenAI-Compatible API**: Drop-in replacement for OpenAI's Chat Completions API
- ✅ **Hugging Face Integration**: Works with any Hugging Face transformer model
- ✅ **Performance Benchmarking**: Built-in tool for measuring latency and throughput
- ✅ **GPU Support**: Automatic CUDA detection and optimization
- ✅ **Quantization Support**: 4-bit and 8-bit quantization options for memory efficiency
- ✅ **Chat Templates**: Automatic chat template detection for better prompt formatting

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start the API Server

By default, the server will load GPT-2. To use a different model, set the `HF_MODEL_NAME` environment variable:

```bash
export HF_MODEL_NAME="microsoft/DialoGPT-medium"  # or any other model
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 2. Test the API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a haiku about AI."}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 3. Run Benchmarking

```bash
python benchmark.py --prompt "Explain quantum computing in simple terms" --runs 10
```

## API Endpoints

### POST `/v1/chat/completions`

Create a chat completion (OpenAI-compatible).

**Request:**
```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": null,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1694268190,
  "model": "default",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  }
}
```

### GET `/v1/models`

List available models.

### GET `/`

Health check endpoint.

## Benchmarking Tool

The `benchmark.py` script provides comprehensive performance metrics:

```bash
python benchmark.py \
  --prompt "Your prompt here" \
  --runs 10 \
  --url http://localhost:8000 \
  --model default \
  --temperature 0.7 \
  --max-tokens 200 \
  --output results.json
```

**Options:**
- `--prompt`: The prompt to benchmark (required)
- `--runs`: Number of requests to make (default: 5)
- `--url`: API base URL (default: http://localhost:8000)
- `--model`: Model identifier (default: default)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max-tokens`: Maximum tokens to generate
- `--output`: Save results to JSON file

**Metrics Provided:**
- Response time (mean, median, min, max, stddev)
- Tokens per second (mean, median, min, max, stddev)
- Token counts (prompt, completion, total)
- Sample response text

## Using with Different Models

### Small Models (for testing)
```bash
export HF_MODEL_NAME="gpt2"
python app.py
```

### Medium Models
```bash
export HF_MODEL_NAME="microsoft/DialoGPT-medium"
python app.py
```

### Large Models (requires sufficient GPU memory)
```bash
export HF_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
python app.py
```

### With Quantization (4-bit for memory efficiency)
Edit `model_manager.py` and modify the `load_default_model` method to use:
```python
self.load_model(default_model, load_in_4bit=True)
```

## Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "default",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## Environment Variables

- `HF_MODEL_NAME`: Hugging Face model identifier (default: "gpt2")

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for larger models)
- At least 4GB RAM (more for larger models)

## License

This project is provided as-is for use with Hugging Face Transformers models.

