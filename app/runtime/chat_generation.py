"""What a chat request asks the engine to do: prompt tokens, sampling options, images, and the
structured-output constraint; plus the end-of-reply tokens and an already-open reasoning block."""

import torch

from app.models.handle import ModelHandle
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.json_grammar import JsonTokenConstraint
from app.runtime.qwen3vl_fusion import FusedPrompt
from app.runtime.thinking_splitter import MARKERS as THINKING_MARKERS
from app.schemas.chat import ChatRequest


class ChatGeneration:
    @staticmethod
    def eos_ids(handle: ModelHandle) -> set[int]:
        return {handle.tokenizer.eos_token_id} | handle.extra_eos_token_ids

    @staticmethod
    def thinking_until(handle: ModelHandle, token_ids: list[int]) -> str | None:
        """The closing marker when the prompt already opened a reasoning block."""
        prompt_end = handle.tokenizer.decode(token_ids[-4:]).rstrip()
        return next((c for o, c in THINKING_MARKERS if prompt_end.endswith(o)), None)

    def generation_request(
        self,
        request: ChatRequest,
        handle: ModelHandle,
        prompt_token_ids: list[int],
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
        fused: FusedPrompt | None = None,
    ) -> GenerationRequest:
        options = request.options
        schema = request.format if isinstance(request.format, dict) else None
        constraint = (
            JsonTokenConstraint(handle.tokenizer, self.eos_ids(handle), schema)
            if request.format is not None
            else None
        )
        return GenerationRequest(
            input_ids=torch.tensor([prompt_token_ids], dtype=torch.long),
            image_embeddings=image_embeddings,
            position_ids=fused.position_ids if fused else None,
            position_delta=fused.position_delta if fused else 0,
            cache_tag=request.cache_tag or "",
            sampling=SamplingConfig(
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                repeat_penalty=options.repeat_penalty,
                num_ctx=options.num_ctx,
                num_predict=options.num_predict,
                seed=options.seed,
                min_p=options.min_p,
                presence_penalty=options.presence_penalty,
                frequency_penalty=options.frequency_penalty,
            ),
            constraint=constraint,
        )
