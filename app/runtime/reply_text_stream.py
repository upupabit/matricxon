"""Turns a reply's generated token ids into the visible text plus any tool calls: decode -> split
off the reasoning (`<think>...</think>`) -> hold back tool-call markup -> drop stray special
tokens."""

from app.runtime.special_token_filter import SpecialTokenTextFilter
from app.runtime.thinking_splitter import ThinkingSplitter
from app.runtime.tokenizer import IncrementalTextDecoder
from app.runtime.tool_call_formats import ToolCallFormat
from app.runtime.tool_call_scanner import ToolCallScanner
from app.schemas.chat import ToolCall


class ReplyTextStream:
    def __init__(
        self,
        tokenizer: object,
        call_format: ToolCallFormat | None = None,
        tools: list[dict] | None = None,
        thinking_until: str | None = None,
    ) -> None:
        self._decoder = IncrementalTextDecoder(tokenizer)
        # The closing marker when the prompt already opened a reasoning block (ThinkingSplitter).
        self._splitter = ThinkingSplitter(thinking_until)
        self._filter = SpecialTokenTextFilter()
        self._thinking_filter = SpecialTokenTextFilter()
        # Scan only when the request offered tools and the model has a known call format.
        self._scanner = ToolCallScanner(call_format, tools) if call_format and tools else None

    def push(self, token_id: int) -> tuple[str, str]:
        """(reasoning, answer) text released by this token."""
        thinking, text = self._splitter.push(self._decoder.push(token_id))
        return self._thinking_filter.feed(thinking), self._answer(text)

    def finish(self) -> tuple[str, str, list[ToolCall]]:
        """The last reasoning and answer text, and the tool calls found in the reply."""
        thinking, text = self._splitter.finish()
        text = self._answer(text)
        tail, calls = self._scanner.finish() if self._scanner else ("", [])
        thinking = self._thinking_filter.feed(thinking) + self._thinking_filter.flush()
        return thinking, text + self._filter.feed(tail) + self._filter.flush(), calls

    def _answer(self, text: str) -> str:
        if self._scanner is not None:
            text = self._scanner.feed(text)
        return self._filter.feed(text)
