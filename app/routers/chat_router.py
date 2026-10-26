import asyncio
import logging
import time
from collections.abc import AsyncIterator

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.manager import ModelManager
from app.native.gemm import NativeGemm
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.live_requests import LiveRegistry, LiveRequest
from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.runtime.qwen3vl_fusion import FusedPrompt, build_qwen3vl_prompt
from app.runtime.reply_text_stream import ReplyTextStream
from app.runtime.rerank_engine import ModelKind
from app.runtime.thinking_splitter import MARKERS as THINKING_MARKERS
from app.runtime.vision_fusion import build_prompt_with_images
from app.schemas.chat import ChatChunk, ChatDoneChunk, ChatRequest, ChatStreamMessage, ToolCall
from app.server.errors import PromptTooLongError
from app.server.ndjson import NDJSONResponse
from app.server.node_kind import require_compute

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequestHandler:
    """Handles `POST /api/chat`.

    All fallible work (model load, prompt building, tokenization, num_ctx validation) happens
    synchronously in `handle`, before any `NDJSONResponse` exists: Starlette sends status/headers
    before pulling a streaming response's first item, so a validation error raised here becomes a
    clean pre-stream HTTP error (via the `MatricxonError` -> `JSONResponse` handler), not a broken
    in-stream one.
    """

    def __init__(self, model_manager: ModelManager, catalog: ModelCatalog) -> None:
        self._model_manager = model_manager
        self._catalog = catalog
        self._prompt_builder = Mistral3PromptBuilder()

    def handle(self, request: ChatRequest) -> JSONResponse | NDJSONResponse:
        if request.is_unload_call():
            if request.request_id is not None:
                # Stops only that reply; the model and every other reply are left alone.
                self._model_manager.cancel_request(request.model, request.request_id)
            else:
                self._model_manager.unload(request.model)
            return JSONResponse(status_code=200, content={})

        live = LiveRegistry.shared().start(request.request_id, request.model)
        try:
            response = self._handle(request, live)
        except BaseException:
            live.finish()
            raise
        if not isinstance(response, NDJSONResponse):
            live.finish()
        return response

    def _handle(self, request: ChatRequest, live: LiveRequest) -> JSONResponse | NDJSONResponse:
        load_started = time.monotonic()
        handle = self._model_manager.get_or_load(
            request.model, keep_alive_seconds=request.keep_alive, role=request.model_role
        )
        ModelKind.require_generative(handle)
        load_duration = time.monotonic() - load_started
        live.loaded()
        logger.info("model %r ready in %.2fs (get_or_load)", request.model, load_duration)

        # A stop (unload) for this model arrived while it had no handle, i.e. while the
        # get_or_load above was still in loading/dequantization, which alone can outlast a
        # caller's patience. See ModelManager.pop_cancelled_load: without this check the stop would
        # be lost when loading finished and generation would start anyway.
        if self._model_manager.pop_cancelled_load(request.model):
            return JSONResponse(status_code=200, content={})
        if request.request_id is not None and self._model_manager.pop_cancelled_request(
            request.request_id
        ):
            return JSONResponse(status_code=200, content={})

        stage_started = time.monotonic()
        # The model's chat template (app.runtime.chat_template): Mistral3PromptBuilder only for
        # mistral3 itself or a model with no template.
        prompt_builder = handle.prompt_builder or self._prompt_builder
        prompt = prompt_builder.build(request.messages, request.tools, request.think)
        logger.info(
            "prompt built: %d messages -> %d chars in %.1fms",
            len(request.messages),
            len(prompt),
            (time.monotonic() - stage_started) * 1000,
        )

        self._log_tools_offered(request, handle)

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

        generation_request = self._build_generation_request(
            request, prompt_token_ids, image_embeddings, fused
        )
        return NDJSONResponse(
            self._stream(
                handle, generation_request, len(prompt_token_ids), load_duration, request, live
            )
        )

    @staticmethod
    def _log_tools_offered(request: ChatRequest, handle: ModelHandle) -> None:
        """Tool-use trace: level 1 says whether tools were offered, level 2 which."""
        if not request.tools:
            return
        names = [(t.get("function") or {}).get("name", "?") for t in request.tools]
        fmt = type(handle.tool_call_format).__name__ if handle.tool_call_format else "none"
        returned = sum(1 for m in request.messages if m.role == "tool")
        logger.info(
            "tools: %d offered, call format %s, %d tool results in the history",
            len(names),
            fmt,
            returned,
        )
        logger.debug("tools offered: %s", ", ".join(names))

    @staticmethod
    def _log_stop(token_id: int, handle: ModelHandle, request: ChatRequest) -> None:
        """Why a reply ended: the stop token, and whether it is the model's own end token."""
        text = handle.tokenizer.decode([token_id])
        own = token_id == handle.tokenizer.eos_token_id
        logger.info(
            "stopped on token %d %r (%s)",
            token_id,
            text,
            "end of turn" if own else "extra stop token",
        )
        if request.tools:
            logger.debug(
                "stop token seen while tools were on - a tool call ends with <|tool_response>"
            )

    @staticmethod
    def _log_tool_result(request: ChatRequest, tail: str, tool_calls: list[ToolCall]) -> None:
        """What the reply turned into: tool calls, or plain text (no call made)."""
        if not request.tools:
            return
        if tool_calls:
            names = ", ".join(c.function.name for c in tool_calls)
            logger.info("tool calls parsed: %d (%s)", len(tool_calls), names)
            for call in tool_calls:
                logger.debug(
                    "tool call %s arguments: %s", call.function.name, call.function.arguments
                )
        else:
            logger.info("no tool call in the reply (%d chars of text)", len(tail))
            logger.debug("reply text without a tool call: %.300s", tail)

    def _build_generation_request(
        self,
        request: ChatRequest,
        prompt_token_ids: list[int],
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
        fused: FusedPrompt | None = None,
    ) -> GenerationRequest:
        options = request.options
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
            ),
        )

    async def _stream(
        self,
        handle: ModelHandle,
        generation_request: GenerationRequest,
        prompt_eval_count: int,
        load_duration: float,
        request: ChatRequest,
        live: LiveRequest,
    ) -> AsyncIterator[dict]:
        """Wraps `_stream_tokens` so the live entry ends however the stream does."""
        live.queued(prompt_eval_count)
        try:
            async for chunk in self._stream_tokens(
                handle, generation_request, prompt_eval_count, load_duration, request, live
            ):
                yield chunk
        finally:
            live.finish()

    async def _stream_tokens(
        self,
        handle: ModelHandle,
        generation_request: GenerationRequest,
        prompt_eval_count: int,
        load_duration: float,
        request: ChatRequest,
        live: LiveRequest,
    ) -> AsyncIterator[dict]:
        request_id = request.request_id
        eos_token_ids = {handle.tokenizer.eos_token_id} | handle.extra_eos_token_ids
        engine = ChatEngine(
            handle.architecture,
            eos_token_ids=eos_token_ids,
            prompt_cache=handle.prompt_cache,
            live=live,
        )
        # Thinking on: the prompt may already open the block, so the reply starts as reasoning.
        prompt_end = handle.tokenizer.decode(generation_request.input_ids[0, -4:].tolist()).rstrip()
        until = next((c for o, c in THINKING_MARKERS if prompt_end.endswith(o)), None)
        reply = ReplyTextStream(handle.tokenizer, handle.tool_call_format, request.tools, until)
        # Cancelled in the gap between handle and this stream starting: nothing to run.
        if request_id is not None and self._model_manager.pop_cancelled_request(request_id):
            return
        stop_check = handle.worker.stop_check_for(request_id)
        results = handle.worker.stream(
            lambda: engine.stream_steps(generation_request, stop_check=stop_check),
            request_id=request_id,
        )

        eval_count = 0
        generation_started = time.monotonic()
        last_token_at = generation_started
        while True:
            item = await results.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            eval_count += 1
            now = time.monotonic()
            logger.info("token %d: id=%d elapsed=%.3fs", eval_count, item, now - last_token_at)
            last_token_at = now
            # The EOS token is a counted generation step (as ChatEngine's token_ids/eval_count) but
            # its decoded text (e.g. "</s>") must not reach the visible message; `done: true` tells
            # a client generation ended.
            if item in eos_token_ids:
                self._log_stop(item, handle, request)
                continue
            thinking, text = reply.push(item)
            if thinking or text:
                message = ChatStreamMessage(content=text, thinking=thinking or None)
                yield ChatChunk(message=message).to_ndjson_dict()

        thinking, tail, tool_calls = reply.finish()
        self._log_tool_result(request, tail, tool_calls)
        if thinking or tail or tool_calls:
            message = ChatStreamMessage(
                content=tail, thinking=thinking or None, tool_calls=tool_calls or None
            )
            yield ChatChunk(message=message).to_ndjson_dict()

        eval_duration = time.monotonic() - generation_started
        logger.info(
            "generation done: %d tokens in %.2fs (%.2fs/token)",
            eval_count,
            eval_duration,
            eval_duration / eval_count if eval_count else 0.0,
        )
        native = NativeGemm.active()
        if native is not None:
            calls, seconds = native.take_stats()
            logger.info(
                "native kernels: %d calls, %.2fs (%.0f%% of generation)",
                calls,
                seconds,
                100 * seconds / eval_duration if eval_duration else 0.0,
            )
        first_token = live.first_token_seconds()
        yield ChatDoneChunk(
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            load_duration=_seconds_to_ns(load_duration),
            prompt_eval_duration=_seconds_to_ns(first_token) if first_token is not None else None,
            eval_duration=_seconds_to_ns(eval_duration),
            total_duration=_seconds_to_ns(load_duration + eval_duration),
        ).to_ndjson_dict()


def _seconds_to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


@router.post("/api/chat", dependencies=[Depends(require_compute)], response_model=None)
async def post_chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    catalog: ModelCatalog = Depends(get_model_catalog),
) -> JSONResponse | NDJSONResponse:
    # Offloaded to a thread: model loading (seconds of blocking disk/dequant work) and the rest of
    # ChatRequestHandler.handle must not block the event loop, which keeps answering /api/tags
    # health probes during a chat.
    handler = ChatRequestHandler(model_manager, catalog)
    return await asyncio.to_thread(handler.handle, request)
