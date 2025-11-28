# autointerp_steer/llm_clients.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import requests
import os


@dataclass
class VLLMHTTPConfig:
    base_url: str          # e.g. "http://localhost:8002/v1"
    model: str             # e.g. "Qwen/Qwen2.5-72B-Instruct"
    api_key: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: int = 60      # seconds


class VLLMHTTPClient:
    """
    Very small OpenAI-compatible HTTP client for vLLM.
    Expects a /chat/completions endpoint.
    
    Provides both sync (.chat()) and async (.generate()) interfaces
    for compatibility with existing code.
    """
    def __init__(self, cfg: VLLMHTTPConfig):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.session = requests.Session()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Synchronous chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        
        Returns:
            Response text string
        """
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        headers = {"Content-Type": "application/json"}
        api_key = self.cfg.api_key or os.getenv("VLLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def generate(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """
        Async wrapper for chat completion (for compatibility with existing code).
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Optional override for max_tokens (uses config default if None)
            temperature: Optional override for temperature (uses config default if None)
        
        Returns:
            Response text string
        """
        import asyncio
        
        # Create a temporary payload with overrides
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
        }
        headers = {"Content-Type": "application/json"}
        api_key = self.cfg.api_key or os.getenv("VLLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Run the HTTP request in a thread pool to avoid blocking
        def _make_request():
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.cfg.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _make_request)
        return response

