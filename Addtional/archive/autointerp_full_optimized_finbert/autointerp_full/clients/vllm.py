import json
from asyncio import sleep
import httpx
from autointerp_full import logger
from .client import Client, Response
from .types import ChatFormatRequest

class VLLMClient(Client):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8002/v1",
                max_tokens: int = 1000,
        temperature: float = 1.0,
    ):
        super().__init__(model)
        
        # Ensure base_url doesn't end with /chat/completions
        if base_url.endswith("/chat/completions"):
            base_url = base_url.replace("/chat/completions", "")
        if base_url.endswith("/v1"):
            self.url = f"{base_url}/chat/completions"
        else:
            self.url = f"{base_url}/v1/chat/completions"
            
        self.max_tokens = max_tokens
        self.temperature = temperature
        timeout_config = httpx.Timeout(30.0)  # Increased timeout
        self.client = httpx.AsyncClient(timeout=timeout_config)
        logger.info(f"VLLM Client initialized with base URL: {self.url}")

    def postprocess(self, response):
        response_json = response.json()
        msg = response_json["choices"][0]["message"]["content"]
        return Response(text=msg)

    async def generate(self, prompt: ChatFormatRequest, max_retries: int = 3, **kwargs):  # Increased retries
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        
        data = {
            "model": self.model,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(max_retries):
            try:
                response = await self.client.post(url=self.url, json=data, timeout=300)  # Increased request timeout
                response.raise_for_status()  # Raise exception for non-200 status codes
                result = self.postprocess(response)
                return result
            except httpx.HTTPStatusError as e:
                logger.warning(f"Attempt {attempt + 1}: HTTP {e.response.status_code}: {e.response.text[:200]}, retrying...")
                await sleep(2)  # Increased delay
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {repr(e)}, retrying...")
                await sleep(2)  # Increased delay

        raise RuntimeError("Failed to generate text after multiple attempts.")
