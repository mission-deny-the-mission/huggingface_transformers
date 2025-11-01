"""
OpenAI-compatible API wrapper for Hugging Face Transformers.
"""
import os
import time
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

