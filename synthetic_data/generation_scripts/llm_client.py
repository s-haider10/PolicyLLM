#!/usr/bin/env python3
"""
Local LLM client for synthetic data generation.
Supports Phi-2, Llama-3, and other HuggingFace models with GPU acceleration.
"""
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Phi2Client:
    def __init__(
        self,
        model_path: str = "microsoft/phi-2",
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        cache_dir: str = "/scratch2/shared_models",
    ) -> None:
        """
        Initialize LLM client with local model.
        
        Args:
            model_path: Model name or path ('microsoft/phi-2', 'meta-llama/Meta-Llama-3-8B', etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Maximum number of tokens to generate
            device: Device to run on ('cuda:0', 'cuda:1', etc). If None, auto-detect.
            cache_dir: HuggingFace cache directory with downloaded models
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.is_llama = "llama" in model_path.lower()
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                # Use CUDA_VISIBLE_DEVICES if set, otherwise use cuda:0
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                device_id = cuda_visible.split(',')[0]
                self.device = f"cuda:{device_id}" if device_id.isdigit() else "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        model_name = model_path.split('/')[-1]
        print(f"Loading {model_name} from cache on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        
        # Set pad token if not set (needed for Llama models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype for GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map=self.device,
        )
        self.model.eval()
        
        print(f"✓ {model_name} loaded successfully on {self.device}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: The prompt to complete
            system: Optional system message (prepended to prompt if provided)
            
        Returns:
            Generated text (prompt excluded and cleaned)
        """
        # Construct full prompt - use simple format that works across models
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(full_prompt):].strip()
        
        # Clean up repetition - take only first non-empty line if there's excessive repetition
        lines = generated_text.split('\n')
        if lines:
            generated_text = lines[0].strip()
        
        # Remove common truncation artifacts
        if not generated_text or len(generated_text) < 10:
            raise RuntimeError(f"{self.model_path} returned an empty or too short response.")
        
        return generated_text

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
