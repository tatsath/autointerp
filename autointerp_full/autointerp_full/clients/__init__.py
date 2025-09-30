from .client import Client
from .openrouter import OpenRouter
from .transformers_client import TransformersClient
from .transformers_fast_client import TransformersFastClient

# Only import Offline if vllm is available
try:
    from .offline import Offline
    __all__ = ["Client", "OpenRouter", "Offline", "TransformersClient", "TransformersFastClient"]
except ImportError:
    __all__ = ["Client", "OpenRouter", "TransformersClient", "TransformersFastClient"]

# Try to import ExLlamaV2 client
try:
    from .exllamav2_client import ExLlamaV2Client
    __all__.append("ExLlamaV2Client")
except ImportError:
    pass
