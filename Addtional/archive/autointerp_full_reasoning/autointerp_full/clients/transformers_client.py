import asyncio
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from autointerp_full import logger

from .client import Client, Response


@dataclass
class Top_Logprob:
    token: str
    logprob: float


@dataclass
class Logprobs:
    token: str
    top_logprobs: list[Top_Logprob]


@dataclass
class Statistics:
    num_prompt_tokens: int
    num_new_tokens: int
    num_generated_tokens: int


class TransformersClient(Client):
    provider = "transformers"

    def __init__(
        self,
        model: str,
        max_memory: float = 0.85,
        batch_size: int = 1,
        max_model_len: int = 2048,
        number_tokens_to_generate: int = 500,
        num_gpus: int = 1,
        **kwargs,
    ):
        super().__init__(model)
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.number_tokens_to_generate = number_tokens_to_generate
        self.num_gpus = num_gpus
        
        # Load model and tokenizer
        logger.info(f"Loading model {model} with Transformers...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully on {self.model.device}")

    async def generate(
        self, prompt: Union[str, list[dict[str, str]]], **kwargs
    ) -> str | Response:
        """Generate text using regular Transformers (no VLLM, no KV cache)"""
        
        # Convert prompt to string if it's a list
        if isinstance(prompt, list):
            prompt_text = ""
            for item in prompt:
                if item.get("role") == "user":
                    prompt_text += f"User: {item['content']}\n"
                elif item.get("role") == "assistant":
                    prompt_text += f"Assistant: {item['content']}\n"
        else:
            prompt_text = prompt
            
        # Truncate prompt if too long
        if len(prompt_text) > self.max_model_len:
            prompt_text = prompt_text[-self.max_model_len:]
            
        # Tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len
        ).to(self.model.device)
        
        # Generate without KV cache (regular inference)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.number_tokens_to_generate,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable KV cache!
            )
        
        # Decode response
        response_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up memory
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return response_text

    def __del__(self):
        """Clean up when client is destroyed"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()




















