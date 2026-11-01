"""What a chat reply did, for the log: tools offered, why it stopped, tool calls found."""

import logging

from app.models.handle import ModelHandle
from app.native.gemm import NativeGemm
from app.schemas.chat import ChatRequest, ToolCall

logger = logging.getLogger("app.routers.chat_router")


class ChatLog:
    @staticmethod
    def tools_offered(request: ChatRequest, handle: ModelHandle) -> None:
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
    def stop(token_id: int, handle: ModelHandle, request: ChatRequest) -> None:
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
    def tool_result(request: ChatRequest, tail: str, tool_calls: list[ToolCall]) -> None:
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

    @staticmethod
    def generation_done(eval_count: int, eval_duration: float) -> None:
        """Speed of the reply, and the native kernels' share of it."""
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
