"""
OpenAI-compatible API wrapper for Hugging Face Transformers.
"""
import os
import time
import argparse
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from model_manager import ModelManager

# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_manager.load_default_model()
    yield
    # Cleanup on shutdown
    model_manager.unload_model()


app = FastAPI(
    title="Hugging Face Transformers OpenAI-Compatible API",
    description="OpenAI-compatible API wrapper for Hugging Face Transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models matching OpenAI API format
class Message(BaseModel):
    role: str = Field(..., description="The role of the message (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default", description="Model identifier")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream responses")


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Hugging Face Transformers OpenAI-Compatible API"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_manager.current_model_name or "default",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "huggingface",
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (OpenAI-compatible endpoint)."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    
    # Get the model
    generator = model_manager.get_generator()
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format messages for the model
    prompt = model_manager.format_messages(request.messages)
    
    # Generate response
    start_time = time.time()
    
    generation_kwargs = {
        "temperature": request.temperature if request.temperature > 0 else None,
        "top_p": request.top_p if request.top_p < 1.0 else None,
        "max_new_tokens": request.max_tokens or 512,
        "do_sample": request.temperature > 0,
        "return_full_text": False,  # Only return the generated part
    }
    
    # Handle stop sequences - transformers pipeline handles these via stopping criteria
    # For now, we'll process them after generation if needed
    
    try:
        output = generator(prompt, **generation_kwargs)
        generated_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
        
        # Extract only the newly generated part (should already be just the new part with return_full_text=False)
        if prompt in generated_text:
            completion_text = generated_text[len(prompt):].strip()
        else:
            completion_text = generated_text.strip()
        
        # Handle stop sequences
        if request.stop:
            for stop_seq in request.stop:
                if stop_seq in completion_text:
                    completion_text = completion_text.split(stop_seq)[0].strip()
                    break
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
    
    generation_time = time.time() - start_time
    
    # Estimate token counts (rough approximation)
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion_text.split())
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible API wrapper for Hugging Face Transformers")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to load (e.g., 'LiquidAI/LFM2-700M', 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode even if GPU is available"
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Number of CPU threads to use for inference"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model presets and exit"
    )
    return parser.parse_args()


def list_available_models():
    """List available model presets."""
    models = {
        "default": "LiquidAI/LFM2-700M",
        "qwen3": "Qwen/Qwen3-7B-Instruct",
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
    
    print("Available model presets:")
    print("=" * 50)
    for name, model_id in models.items():
        print(f"{name:12} -> {model_id}")
    print("=" * 50)
    print("Usage examples:")
    print("  python app.py --model qwen")
    print("  python app.py --model Qwen/Qwen2.5-7B-Instruct")
    print("  python app.py --model gpt2 --cpu")


if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    
    # Handle list models command
    if args.list_models:
        list_available_models()
        exit(0)
    
    # Set environment variables based on command line arguments
    if args.model:
        os.environ["HF_MODEL_NAME"] = args.model
    
    if args.cpu:
        os.environ["FORCE_CPU"] = "true"
    
    if args.cpu_threads:
        os.environ["CPU_THREADS"] = str(args.cpu_threads)
    
    # Update model manager with the specified model if provided
    if args.model:
        model_manager.default_model = args.model
    
    uvicorn.run(app, host=args.host, port=args.port)

