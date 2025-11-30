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
        max_tokens: int = 2000,  # Increased from 1000 to handle longer responses
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
        # Set timeout - use a single timeout value for simplicity
        # Previous: 30s was too short, now using 600s for large prompts
        timeout_config = httpx.Timeout(600.0)  # 600 seconds total timeout
        self.client = httpx.AsyncClient(timeout=timeout_config)
        logger.info(f"VLLM Client initialized with base URL: {self.url}, max_tokens={max_tokens}, timeout=600s")
        
        # Test connection to vLLM server (non-blocking, just log warning if fails)
        try:
            # Try to check if server is reachable
            test_url = base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
            test_url = test_url.replace("/chat/completions", "")
            if not test_url.endswith("/v1"):
                test_url = f"{test_url}/v1"
            test_url = f"{test_url}/models"
            
            # Use sync client for quick connection test
            import requests
            try:
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ vLLM server connection test successful at {test_url}")
                else:
                    logger.warning(f"⚠ vLLM server returned status {response.status_code} at {test_url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠ Could not connect to vLLM server at {test_url}: {e}")
                logger.warning("   Explanation generation may fail. Make sure vLLM server is running.")
        except Exception as e:
            logger.warning(f"⚠ Could not test vLLM server connection: {e}")

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

        # Estimate prompt size to log
        prompt_size = len(json.dumps(data))
        logger.debug(f"vLLM request size: ~{prompt_size // 4} tokens (estimated), max_tokens={max_tokens}")
        
        for attempt in range(max_retries):
            try:
                # Remove timeout parameter - use client-level timeout instead
                response = await self.client.post(url=self.url, json=data)
                response.raise_for_status()  # Raise exception for non-200 status codes
                result = self.postprocess(response)
                return result
            except httpx.HTTPStatusError as e:
                error_text = e.response.text[:500] if hasattr(e.response, 'text') else str(e.response)
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: HTTP {e.response.status_code}: {error_text}")
                if e.response.status_code == 413:  # Payload too large
                    logger.error(f"Prompt too large! Consider reducing k_max_act or window size.")
                elif e.response.status_code == 429:  # Too many requests
                    logger.warning("Rate limited, waiting longer...")
                    await sleep(5)
                else:
                    await sleep(2)
            except httpx.TimeoutException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Request timeout after 600s. Prompt may be too large or server overloaded.")
                await sleep(5)  # Wait longer for timeout errors
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
                # Connection errors - server might be down or unreachable
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Connection error: {type(e).__name__}: {repr(e)}")
                logger.warning(f"Check if vLLM server is running at {self.url}")
                if attempt == max_retries - 1:
                    # Last attempt - provide more details
                    logger.error(f"Final connection attempt failed. Server may be down or unreachable at {self.url}")
                await sleep(2)
            except httpx.RequestError as e:
                # Other request errors
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Request error: {type(e).__name__}: {repr(e)}")
                await sleep(2)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {repr(e)}")
                await sleep(2)

        # Provide more informative error message based on last exception type
        raise RuntimeError("Failed to generate text after multiple attempts. Check vLLM server connection.")
