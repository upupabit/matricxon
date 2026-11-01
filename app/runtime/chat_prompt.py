"""A chat request's prompt as token ids: the model's chat template, then images (vision models)
and tokenizing, with the prompt checked against num_ctx."""

import logging
import time
from dataclasses import dataclass

import torch

from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.runtime.chat_log import ChatLog
from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.runtime.qwen3vl_fusion import FusedPrompt, build_qwen3vl_prompt
from app.runtime.vision_fusion import build_prompt_with_images
from app.schemas.chat import ChatRequest
from app.server.errors import PromptTooLongError

logger = logging.getLogger("app.routers.chat_router")


@dataclass
class PromptTokens:
    token_ids: list[int]
    images: list[tuple[int, torch.Tensor]] | None
    fused: FusedPrompt | None


class ChatPrompt:
    def __init__(self, catalog: ModelCatalog, default_builder: Mistral3PromptBuilder) -> None:
        self._catalog = catalog
        self._default_builder = default_builder

    def build(self, handle: ModelHandle, request: ChatRequest) -> PromptTokens:
        stage_started = time.monotonic()
        # The model's chat template (app.runtime.chat_template): Mistral3PromptBuilder only for
        # mistral3 itself or a model with no template.
        prompt_builder = handle.prompt_builder or self._default_builder
        prompt = prompt_builder.build(request.messages, request.tools, request.think)
        logger.info(
            "prompt built: %d messages -> %d chars in %.1fms",
            len(request.messages),
            len(prompt),
            (time.monotonic() - stage_started) * 1000,
        )

        ChatLog.tools_offered(request, handle)

        # Vision support (LlamaArchitecture only, e.g. LLaVA): app.runtime.vision_fusion. A request
        # with images but no paired mmproj (ModelCatalog.find_paired_mmproj) falls through to the
        # plain tokenizer.encode path: matricxon's original "count and ignore" v1 stance for
        # architectures not wired for fusion.
        all_images = [img for message in request.messages for img in (message.images or [])]
        mmproj = self._catalog.find_paired_mmproj(request.model) if all_images else None

        stage_started = time.monotonic()
        image_embeddings = None
        fused = None
        if mmproj is not None and handle.architecture.NAME == "qwen35":
            fused = build_qwen3vl_prompt(
                prompt, all_images, handle.tokenizer, mmproj, handle.architecture.device
            )
            prompt_token_ids, image_embeddings = fused.token_ids, fused.image_embeddings
        elif mmproj is not None:
            prompt_token_ids, image_embeddings = build_prompt_with_images(
                prompt, all_images, handle.tokenizer, mmproj, handle.architecture.device
            )
        else:
            prompt_token_ids = handle.tokenizer.encode(
                prompt, add_bos=prompt_builder.wants_bos(prompt)
            )
        logger.info(
            "tokenized: %d chars -> %d tokens in %.1fms (num_ctx=%d, images=%d)",
            len(prompt),
            len(prompt_token_ids),
            (time.monotonic() - stage_started) * 1000,
            request.options.num_ctx,
            len(all_images),
        )
        if len(prompt_token_ids) > request.options.num_ctx:
            raise PromptTooLongError(
                f"prompt has {len(prompt_token_ids)} tokens, "
                f"exceeds num_ctx={request.options.num_ctx}"
            )
        return PromptTokens(prompt_token_ids, image_embeddings, fused)
