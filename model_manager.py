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
        self.default_model = os.getenv("HF_MODEL_NAME", "LiquidAI/LFM2-700M")
        
        # Check for forced CPU mode via environment variable
        force_cpu = os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes", "on")
        
        if force_cpu:
            self.device = "cpu"
            print("FORCE_CPU enabled: Running in CPU-only mode (GPU disabled)")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            print("Running on CPU. Note: Inference will be slower than GPU.")
            # Set CPU thread count for better performance (configurable via env var)
            cpu_threads = os.getenv("CPU_THREADS")
            if cpu_threads:
                try:
                    num_threads = int(cpu_threads)
                    torch.set_num_threads(num_threads)
                    print(f"CPU threads set to: {num_threads}")
                except ValueError:
                    print(f"Warning: Invalid CPU_THREADS value '{cpu_threads}', using default")
            else:
                # Use PyTorch default (usually number of CPU cores)
                print(f"Using default CPU threads (auto-detected: {torch.get_num_threads()})")
        elif self.device == "cuda":
            print(f"Running on CUDA (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    
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
            
            # Special handling for Qwen3-Next-80B-A3B-Instruct-FP8
            is_qwen3_next_fp8 = "Qwen3-Next-80B-A3B-Instruct-FP8" in model_name
            
            # Load tokenizer (trust_remote_code needed for some models like LFM2-700M and Qwen3-Next)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                # Fallback if trust_remote_code causes issues
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate precision
            # Track if we use device_map so we don't pass device to pipeline
            use_device_map = False
            
            # Quantization only works on CUDA
            if self.device == "cpu" and (load_in_4bit or load_in_8bit):
                print("Warning: Quantization (4-bit/8-bit) is not supported on CPU. Loading full precision model.")
                load_in_4bit = False
                load_in_8bit = False
            
            # Skip additional quantization for FP8 models (already quantized)
            if is_qwen3_next_fp8 and (load_in_4bit or load_in_8bit):
                print("Note: Model is already FP8-quantized, skipping additional quantization.")
                load_in_4bit = False
                load_in_8bit = False
            
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
                    # Special handling for Qwen3-Next-80B-A3B-Instruct-FP8
                    if is_qwen3_next_fp8:
                        print("Loading Qwen3-Next-80B-A3B-Instruct-FP8 with optimized settings...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            trust_remote_code=True,
                            # FP8 models don't need additional quantization
                            # Use optimized settings for MoE models
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                    use_device_map = True
                else:
                    # CPU: use float32, no device_map
                    # Note: Some models may have issues with CPU inference if they require special attention
                    # For maximum compatibility, we load without device_map and let the model stay on CPU
                    if is_qwen3_next_fp8:
                        print("Warning: Qwen3-Next-80B-A3B-Instruct-FP8 requires significant memory. GPU is recommended.")
                        print("Attempting to load on CPU with reduced precision...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,  # Use float16 on CPU to save memory
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,  # Helpful for CPU memory management
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
        # Use the default_model attribute which can be set from command line
        default_model = getattr(self, 'default_model', None) or os.getenv("HF_MODEL_NAME", "LiquidAI/LFM2-700M")
        
        # Map preset names to actual model IDs
        model_presets = {
            "default": "LiquidAI/LFM2-700M",
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2": "Qwen/Qwen2.5-14B-Instruct",
            "qwen3": "Qwen/Qwen2.5-72B-Instruct",
            "qwen3-next": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            "qwen3-next-fp8": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large",
            "dialogpt": "microsoft/DialoGPT-medium",
        }
        
        # Check if the model name is a preset
        if default_model in model_presets:
            actual_model = model_presets[default_model]
            print(f"Using preset '{default_model}' -> {actual_model}")
            default_model = actual_model
        
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

