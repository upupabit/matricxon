"""OpenAI chat requests in, Matricxon chat requests out; Matricxon's reply chunks back as OpenAI's
`chat.completion` (whole) or `chat.completion.chunk`s (streamed)."""

import json
import time
import uuid
from typing import Any

from app.config import settings
from app.openai_compat.schemas import OpenAIChatRequest, OpenAIMessage
from app.schemas.chat import ChatMessage, ChatOptions, ChatRequest, ToolCall, ToolFunctionCall
from app.server.errors import UnsupportedOptionError

_ROLES = {"developer": "system"}  # OpenAI's newer name for the system prompt


class OpenAITranslator:
    def __init__(self, request: OpenAIChatRequest) -> None:
        self.request = request
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())

    # request

    def chat_request(self) -> ChatRequest:
        r = self.request
        if r.n not in (None, 1):
            raise UnsupportedOptionError("n: only one choice per request is supported")
        if r.logprobs:
            raise UnsupportedOptionError("logprobs are not supported")
        given = {
            "temperature": r.temperature,
            "top_p": r.top_p,
            "top_k": r.top_k,
            "min_p": r.min_p,
            "repeat_penalty": r.repeat_penalty,
            "seed": r.seed,
            "presence_penalty": r.presence_penalty,
            "frequency_penalty": r.frequency_penalty,
            "num_predict": r.max_completion_tokens or r.max_tokens,
            "num_ctx": r.num_ctx or settings.openai_num_ctx,
            "stop": [r.stop] if isinstance(r.stop, str) else r.stop,
        }
        options = ChatOptions(**{k: v for k, v in given.items() if v is not None})
        return ChatRequest(
            model=r.model,
            messages=[self._message(m) for m in r.messages],
            tools=None if r.tool_choice == "none" else r.tools,
            options=options,
            think=r.think,
            format=self._format(r.response_format),
        )

    @staticmethod
    def _format(response_format: dict[str, Any] | None) -> str | dict | None:
        kind = (response_format or {}).get("type", "text")
        if kind == "json_object":
            return "json"
        if kind == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            if not isinstance(schema, dict):
                raise UnsupportedOptionError("response_format.json_schema.schema is missing")
            return schema
        if kind != "text":
            raise UnsupportedOptionError(f"response_format type {kind!r} is not supported")
        return None

    @staticmethod
    def _message(message: OpenAIMessage) -> ChatMessage:
        text, images = OpenAITranslator._content(message.content)
        calls = [
            ToolCall(
                id=c.get("id"),
                function=ToolFunctionCall(
                    name=(c.get("function") or {}).get("name", ""),
                    arguments=OpenAITranslator._arguments(
                        (c.get("function") or {}).get("arguments")
                    ),
                ),
            )
            for c in message.tool_calls or []
        ]
        return ChatMessage(
            role=_ROLES.get(message.role, message.role),
            content=text,
            images=images or None,
            tool_calls=calls or None,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )

    @staticmethod
    def _content(content: str | list[dict[str, Any]] | None) -> tuple[str, list[str]]:
        if content is None or isinstance(content, str):
            return content or "", []
        texts, images = [], []
        for part in content:
            if part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url", ""))
                if not url.startswith("data:") or "," not in url:
                    raise UnsupportedOptionError("images must be data: URLs (base64), not links")
                images.append(url.split(",", 1)[1])
        return "\n".join(texts), images

    @staticmethod
    def _arguments(raw: object) -> dict[str, object] | str:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return str(raw)
        return parsed if isinstance(parsed, dict) else str(raw)

    # reply

    @staticmethod
    def tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "index": i,
                "id": c.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": c["function"]["name"],
                    "arguments": json.dumps(c["function"].get("arguments") or {})
                    if not isinstance(c["function"].get("arguments"), str)
                    else c["function"]["arguments"],
                },
            }
            for i, c in enumerate(calls)
        ]

    @staticmethod
    def finish_reason(done: dict[str, Any], had_tool_calls: bool) -> str:
        if had_tool_calls:
            return "tool_calls"
        return "length" if done.get("done_reason") == "length" else "stop"

    @staticmethod
    def usage(done: dict[str, Any]) -> dict[str, int]:
        prompt, completion = done.get("prompt_eval_count") or 0, done.get("eval_count") or 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def chunk(self, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
