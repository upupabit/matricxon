"""`POST /v1/chat/completions`: OpenAI's chat API on top of `/api/chat`'s handler, so models,
caching, tools, images, structured output and the sampling options all work the same way."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.openai_compat.schemas import OpenAIChatRequest, OpenAIChatResponse
from app.openai_compat.translate import OpenAITranslator
from app.routers.chat_router import ChatRequestHandler
from app.server.api_keys import require_api_key
from app.server.error_content import ErrorContent
from app.server.ndjson import NDJSONResponse
from app.server.node_kind import require_compute

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatCompletionsHandler:
    def __init__(self, model_manager: ModelManager, catalog: ModelCatalog) -> None:
        self._handler = ChatRequestHandler(model_manager, catalog)

    async def handle(self, request: OpenAIChatRequest) -> JSONResponse | StreamingResponse:
        translator = OpenAITranslator(request)
        response = await asyncio.to_thread(self._handler.handle, translator.chat_request())
        # Anything else is a reply stopped before it began: an empty answer.
        items = response.items if isinstance(response, NDJSONResponse) else self._nothing()
        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                self._events(translator, items, include_usage), media_type="text/event-stream"
            )
        return JSONResponse((await self._whole(translator, items)).model_dump())

    @staticmethod
    async def _nothing() -> AsyncIterator[dict]:
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop"}

    @staticmethod
    async def _whole(
        translator: OpenAITranslator, items: AsyncIterator[dict]
    ) -> OpenAIChatResponse:
        text, reasoning, calls, done = [], [], [], {}
        async for item in items:
            message = item.get("message") or {}
            text.append(message.get("content") or "")
            reasoning.append(message.get("thinking") or "")
            calls += message.get("tool_calls") or []
            if item.get("done"):
                done = item
        content = "".join(text)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or (None if calls else ""),
        }
        if calls:
            message["tool_calls"] = translator.tool_calls(calls)
        if "".join(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": translator.finish_reason(done, bool(calls)),
        }
        return OpenAIChatResponse(
            id=translator.id,
            created=translator.created,
            model=translator.request.model,
            choices=[choice],
            usage=translator.usage(done),
        )

    @staticmethod
    async def _events(
        translator: OpenAITranslator, items: AsyncIterator[dict], include_usage: bool
    ) -> AsyncIterator[bytes]:
        def event(data: dict[str, Any]) -> bytes:
            return f"data: {json.dumps(data)}\n\n".encode()

        yield event(translator.chunk({"role": "assistant", "content": ""}))
        had_calls = False
        try:
            async for item in items:
                message = item.get("message") or {}
                delta: dict[str, Any] = {}
                if message.get("content"):
                    delta["content"] = message["content"]
                if message.get("thinking"):
                    delta["reasoning_content"] = message["thinking"]
                if message.get("tool_calls"):
                    had_calls = True
                    delta["tool_calls"] = translator.tool_calls(message["tool_calls"])
                if delta:
                    yield event(translator.chunk(delta))
                if item.get("done"):
                    yield event(translator.chunk({}, translator.finish_reason(item, had_calls)))
                    if include_usage:
                        usage = {
                            **translator.chunk({}),
                            "choices": [],
                            "usage": translator.usage(item),
                        }
                        yield event(usage)
        except Exception as error:  # headers are gone: the error travels as an event
            logger.exception("chat completion stream failed")
            yield event(ErrorContent.openai(*ErrorContent.describe(error)))
        yield b"data: [DONE]\n\n"


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_compute), Depends(require_api_key)],
    response_model=None,
)
async def post_chat_completions(
    request: OpenAIChatRequest,
    model_manager: ModelManager = Depends(get_model_manager),
    catalog: ModelCatalog = Depends(get_model_catalog),
) -> JSONResponse | StreamingResponse:
    return await ChatCompletionsHandler(model_manager, catalog).handle(request)
