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

#### Option 1: Using Command Line Arguments (Recommended)

```bash
# Start with default model (LFM2-700M)
python app.py

# Start with a specific model using preset name
python app.py --model qwen

# Start with a specific model using full model ID
python app.py --model Qwen/Qwen2.5-7B-Instruct

# Force CPU mode
python app.py --model gpt2 --cpu

# Specify custom host and port
python app.py --model qwen --host 0.0.0.0 --port 8080

# List available model presets
python app.py --list-models
```

#### Option 2: Using Environment Variables

```bash
# Use a different model
export HF_MODEL_NAME="microsoft/DialoGPT-medium"  # or any other model

# Force CPU-only mode (even if GPU is available)
export FORCE_CPU=true

# Configure CPU threads (optional, for performance tuning)
export CPU_THREADS=4  # Use 4 threads (adjust based on your CPU)

python app.py
```

Or with uvicorn directly:
```bash
FORCE_CPU=true uvicorn app:app --host 0.0.0.0 --port 8000
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

### Available Model Presets

The application includes several model presets for easy access:

```bash
# Default model (LFM2-700M)
python app.py --model default

# Qwen models (recommended for chat)
python app.py --model qwen        # Qwen2.5-7B-Instruct
python app.py --model qwen2       # Qwen2.5-14B-Instruct
python app.py --model qwen3       # Qwen2.5-72B-Instruct
python app.py --model qwen3-next  # Qwen3-Next-80B-A3B-Instruct (MoE model)

# Llama models
python app.py --model llama2-7b   # Llama-2-7b-chat-hf
python app.py --model llama2-13b  # Llama-2-13b-chat-hf
python app.py --model llama3-8b   # Meta-Llama-3-8B-Instruct

# GPT-2 models (good for testing)
python app.py --model gpt2
python app.py --model gpt2-medium
python app.py --model gpt2-large

# Dialog models
python app.py --model dialogpt    # microsoft/DialoGPT-medium
```

### Using Custom Models

You can also use any Hugging Face model by specifying its full ID:

```bash
python app.py --model "meta-llama/Llama-2-70b-chat-hf"
python app.py --model "microsoft/DialoGPT-large"
python app.py --model "Qwen/Qwen3-Next-80B-A3B-Instruct"
```

### CPU Mode

For models that don't require a GPU or when GPU memory is limited:

```bash
python app.py --model gpt2 --cpu --cpu-threads 4
```

### Qwen3-Next Model (FP8 Quantized MoE)

The Qwen3-Next-80B-A3B-Instruct is a special model with these characteristics:
- **Mixture of Experts (MoE)**: 80B total parameters but only ~3B activated per token
- **FP8 Quantization**: Already quantized for memory efficiency
- **Long Context**: Supports input sequences exceeding 260,000 tokens
- **GPU Recommended**: Requires significant memory, GPU is strongly recommended

```bash
# Load the Qwen3-Next model (GPU recommended)
python app.py --model qwen3-next

# Or use the full model name
python app.py --model "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
```

### With Quantization (4-bit for memory efficiency)

Edit `model_manager.py` and modify the `load_default_model` method to use:
```python
self.load_model(default_model, load_in_4bit=True)
```

Note: The Qwen3-Next model is already optimized and doesn't need additional quantization.

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

## Command Line Options

- `--model`: Model name or preset to load (e.g., 'qwen', 'gpt2', or full model ID)
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--cpu`: Force CPU mode even if GPU is available
- `--cpu-threads`: Number of CPU threads to use for inference
- `--list-models`: List available model presets and exit

## Environment Variables

- `HF_MODEL_NAME`: Hugging Face model identifier (default: "LiquidAI/LFM2-700M")
- `FORCE_CPU`: Force CPU-only mode even if GPU is available. Set to `true`, `1`, `yes`, or `on` to enable (default: auto-detect)
- `CPU_THREADS`: Number of CPU threads to use for inference (default: auto-detected based on CPU cores). Set to a number like `4` or `8` to limit threads

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for larger models)
- At least 4GB RAM (more for larger models)

## License

This project is provided as-is for use with Hugging Face Transformers models.

