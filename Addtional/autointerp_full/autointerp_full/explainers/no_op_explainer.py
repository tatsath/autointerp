import asyncio
from abc import ABC
from dataclasses import dataclass

from autointerp_full.explainers.explainer import ExplainerResult
from autointerp_full.latents.latents import ActivatingExample, LatentRecord


@dataclass
class NoOpExplainer(ABC):
    """Doesn't inherit from Explainer due to client being None."""

    client: None = None

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        return ExplainerResult(record=record, explanation="")

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        return []

    def call_sync(self, record):
        return asyncio.run(self.__call__(record))
