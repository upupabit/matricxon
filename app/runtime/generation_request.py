from dataclasses import dataclass
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class SamplingConfig:
    """Field names/defaults match Ollama's `options`. `seed=None` means genuinely random every call,
    never a reused default generator (pAIring's contract for an omitted seed). `num_predict=-1`
    means "generate until EOS or num_ctx", also Ollama's convention."""

    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 2048
    num_predict: int = -1
    seed: int | None = None
    # Off by default. min_p keeps tokens at least this share of the top one's probability; the
    # penalties follow OpenAI: minus `presence` once a token appeared, minus `frequency` per use.
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class TokenConstraint(Protocol):
    """Decides token by token what the reply may contain (the sampler asks before each pick)."""

    def allows(self, token_id: int) -> bool: ...

    def accept(self, token_id: int) -> None: ...


@dataclass(frozen=True)
class GenerationRequest:
    input_ids: torch.Tensor
    sampling: SamplingConfig
    # LLaVA vision support (LlamaArchitecture only; ModelArchitecture.forward): each
    # (start, embeds) pair is an image's projected patch embeddings and the prompt token span (in
    # `input_ids`) they belong at. Relevant only to the prefill forward pass (ChatEngine.stream):
    # the image's tokens are part of the prompt, never regenerated in decode.
    image_embeddings: list[tuple[int, torch.Tensor]] | None = None
    # Qwen3.5 vision (3-axis M-RoPE): per-token (3, prompt_len) positions, and how far decode
    # positions sit from the cache length afterwards (an image spans fewer positions than tokens).
    position_ids: torch.Tensor | None = None
    position_delta: int = 0
    cache_tag: str = ""
    # Limits which tokens may come next (structured output: app.runtime.json_constraint).
    constraint: "TokenConstraint | None" = None


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    finish_reason: Literal["stop", "length"]
