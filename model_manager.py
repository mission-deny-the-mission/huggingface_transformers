"""
Model manager for loading and managing Hugging Face models.
"""
import os
import torch
from typing import Optional, List, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Type for message - can be a dict or Pydantic model
MessageType = Union[Dict[str, str], Any]


class ModelManager:
    """Manages loading and using Hugging Face models."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.current_model_name: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """
        Load a Hugging Face model.
        
        Args:
            model_name: Name of the model to load (e.g., "gpt2", "meta-llama/Llama-2-7b-chat-hf")
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
        """
        try:
            print(f"Loading model: {model_name} on device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate precision
            # Track if we use device_map so we don't pass device to pipeline
            use_device_map = False
            
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                use_device_map = True
            elif load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                use_device_map = True
            else:
                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    use_device_map = True
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                    )
            
            # Create pipeline
            # Don't pass device if we used device_map="auto"
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
            }
            if not use_device_map:
                pipeline_kwargs["device"] = 0 if self.device == "cuda" else -1
            
            self.generator = pipeline("text-generation", **pipeline_kwargs)
            
            self.current_model_name = model_name
            print(f"Model {model_name} loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_default_model(self):
        """Load a default lightweight model."""
        default_model = os.getenv("HF_MODEL_NAME", "LiquidAI/LFM2-700M")
        self.load_model(default_model)
    
    def unload_model(self):
        """Unload the current model from memory."""
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.current_model_name = None
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_generator(self):
        """Get the current generator pipeline."""
        return self.generator
    
    def format_messages(self, messages: List[MessageType]) -> str:
        """
        Format messages into a prompt string.
        
        This is a simple formatting. For chat models, you might want to use
        the tokenizer's chat template if available.
        """
        if not messages:
            return ""
        
        # Helper to extract role/content from message (handles both Pydantic models and dicts)
        def get_msg_attr(msg, attr):
            if hasattr(msg, attr):
                return getattr(msg, attr)
            elif isinstance(msg, dict):
                return msg.get(attr)
            return None
        
        # Try to use tokenizer's chat template if available
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Convert to format expected by apply_chat_template
                formatted_messages = [
                    {"role": get_msg_attr(msg, "role"), "content": get_msg_attr(msg, "content")}
                    for msg in messages
                ]
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        # Fallback to simple formatting
        prompt_parts = []
        for msg in messages:
            role = get_msg_attr(msg, "role")
            content = get_msg_attr(msg, "content")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

