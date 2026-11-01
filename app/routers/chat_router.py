import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.manager import ModelManager
from app.runtime.chat_engine import ChatEngine
from app.runtime.chat_generation import ChatGeneration
from app.runtime.chat_log import ChatLog
from app.runtime.chat_prompt import ChatPrompt
from app.runtime.generation_request import GenerationRequest
from app.runtime.live_requests import LiveRegistry, LiveRequest
from app.runtime.prompt_builder import Mistral3PromptBuilder
from app.runtime.reply_text_stream import ReplyTextStream
from app.runtime.rerank_engine import ModelKind
from app.runtime.stop_sequences import StopSequenceFilter
from app.schemas.chat import ChatChunk, ChatDoneChunk, ChatRequest, ChatStreamMessage
from app.server.api_keys import require_api_key
from app.server.errors import UnsupportedOptionError
from app.server.metrics import Metrics
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

        request = self._prepared(request)
        live = LiveRegistry.shared().start(request.request_id, request.model)
        try:
            response = self._handle(request, live)
        except BaseException:
            live.finish()
            raise
        if not isinstance(response, NDJSONResponse):
            live.finish()
        return response

    @staticmethod
    def _prepared(request: ChatRequest) -> ChatRequest:
        """Structured output turns thinking off; stop texts need an id to stop the reply by."""
        changes: dict[str, object] = {}
        if request.format is not None:
            changes["think"] = False
        if request.options.stop and request.request_id is None:
            changes["request_id"] = f"stop-{uuid.uuid4().hex}"
        return request.model_copy(update=changes) if changes else request

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

        tokens = ChatPrompt(self._catalog, self._prompt_builder).build(handle, request)
        prompt_token_ids, image_embeddings, fused = tokens.token_ids, tokens.images, tokens.fused
        if request.format is not None and ChatGeneration.thinking_until(handle, prompt_token_ids):
            raise UnsupportedOptionError("this model always thinks first: format can't be used")

        generation_request = ChatGeneration().generation_request(
            request, handle, prompt_token_ids, image_embeddings, fused
        )
        return NDJSONResponse(
            self._stream(
                handle, generation_request, len(prompt_token_ids), load_duration, request, live
            )
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
        eos_token_ids = ChatGeneration.eos_ids(handle)
        engine = ChatEngine(
            handle.architecture,
            eos_token_ids=eos_token_ids,
            prompt_cache=handle.prompt_cache,
            live=live,
        )
        # Thinking on: the prompt may already open the block, so the reply starts as reasoning.
        until = ChatGeneration.thinking_until(handle, generation_request.input_ids[0].tolist())
        stops = StopSequenceFilter(request.options.stop)
        ended_by_model = False
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
            if stops.stopped:
                continue  # tokens already on their way when the stop text arrived
            eval_count += 1
            now = time.monotonic()
            logger.info("token %d: id=%d elapsed=%.3fs", eval_count, item, now - last_token_at)
            last_token_at = now
            # The EOS token is a counted generation step (as ChatEngine's token_ids/eval_count) but
            # its decoded text (e.g. "</s>") must not reach the visible message; `done: true` tells
            # a client generation ended.
            if item in eos_token_ids:
                ChatLog.stop(item, handle, request)
                ended_by_model = True
                continue
            thinking, text = reply.push(item)
            text = stops.push(text)
            if stops.stopped and request_id is not None:
                handle.worker.cancel_job(request_id)
            if thinking or text:
                message = ChatStreamMessage(content=text, thinking=thinking or None)
                yield ChatChunk(message=message).to_ndjson_dict()

        thinking, tail, tool_calls = reply.finish()
        tail = stops.push(tail) + stops.finish()
        ChatLog.tool_result(request, tail, tool_calls)
        if thinking or tail or tool_calls:
            message = ChatStreamMessage(
                content=tail, thinking=thinking or None, tool_calls=tool_calls or None
            )
            yield ChatChunk(message=message).to_ndjson_dict()

        eval_duration = time.monotonic() - generation_started
        Metrics.shared().tokens(prompt_eval_count, eval_count)
        ChatLog.generation_done(eval_count, eval_duration)
        first_token = live.first_token_seconds()
        yield ChatDoneChunk(
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            load_duration=_seconds_to_ns(load_duration),
            prompt_eval_duration=_seconds_to_ns(first_token) if first_token is not None else None,
            eval_duration=_seconds_to_ns(eval_duration),
            total_duration=_seconds_to_ns(load_duration + eval_duration),
            done_reason="stop" if ended_by_model or stops.stopped else "length",
        ).to_ndjson_dict()


def _seconds_to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


@router.post(
    "/api/chat",
    dependencies=[Depends(require_compute), Depends(require_api_key)],
    response_model=None,
)
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
