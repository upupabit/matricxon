"""Structured output: a JSON parser that reads a reply byte by byte and says whether it is still on
its way to valid JSON (matching a schema, when one is given), and a token constraint built on it
for the sampler (`ChatRequest.format`: "json" or a JSON Schema).

The parser keeps a stack of open values (app.runtime.json_frames). Checking a candidate token means
copying that small stack and feeding the token's bytes; the sampler only checks the tokens it would
pick (app.runtime.sampler.Sampler._constrained), not the whole vocabulary.
"""

import copy
import re
from typing import Any

from app.runtime.json_frames import (
    CONSUMED,
    ERROR,
    MAX_WHITESPACE,
    POP,
    POP_CONSUMED,
    WHITESPACE,
    LiteralFrame,
    NumberFrame,
    StringFrame,
    ValueFrame,
)

# A special token's text (<|im_end|>, <|eot_id|>...): never part of a JSON reply.
_SPECIAL = re.compile(rb"<\|[^|<>\s]{1,64}\|>")
_TOKEN_FRAMES = (StringFrame, LiteralFrame, NumberFrame)


class JsonGrammar:
    """`format: "json"` asks for an object; a schema decides the root type itself."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        root = schema if schema is not None else {"type": "object"}
        self.stack: list[object] = [ValueFrame(root)]
        self.spaces = 0

    def clone(self) -> "JsonGrammar":
        twin = copy.copy(self)
        twin.stack = [copy.copy(frame) for frame in self.stack]
        return twin

    @property
    def complete(self) -> bool:
        """The root value is finished (a number or literal root counts once it could end)."""
        if not self.stack:
            return True
        return (
            len(self.stack) == 1
            and isinstance(self.stack[0], (LiteralFrame, NumberFrame))
            and self.stack[0].done()
        )

    def feed(self, data: bytes) -> bool:
        """Reads more of the reply; False (and a broken state) as soon as it can't be valid JSON."""
        return all(self._byte(b) for b in data)

    def _byte(self, byte: int) -> bool:
        while True:
            if not self.stack:
                return self._space(byte)
            top = self.stack[-1]
            if not isinstance(top, _TOKEN_FRAMES) and byte in WHITESPACE:
                return self._space(byte)
            self.spaces = 0
            action, result = top.feed(byte)
            if action == ERROR:
                return False
            if action == CONSUMED:
                if result is not None:
                    if isinstance(top, ValueFrame):
                        self.stack[-1] = result  # the value's own frame takes its place
                    else:
                        self.stack.append(result)
                return True
            self.stack.pop()
            if self.stack and hasattr(self.stack[-1], "child_done"):
                self.stack[-1].child_done(result)
            if action == POP_CONSUMED:
                return True
            # POP: the byte ended a number or literal and belongs to the parent: read it again.
            assert action == POP

    def _space(self, byte: int) -> bool:
        # None before the reply's value starts or after it ends: it would only delay the answer.
        at_edge = not self.stack or (len(self.stack) == 1 and isinstance(self.stack[0], ValueFrame))
        if byte not in WHITESPACE or at_edge:
            return False
        self.spaces += 1
        return self.spaces <= MAX_WHITESPACE


class JsonTokenConstraint:
    """Allows only tokens that keep the reply valid JSON; the end-of-reply token only once it is
    complete, and nothing else after that."""

    def __init__(
        self, tokenizer: object, eos_token_ids: set[int], schema: dict[str, Any] | None
    ) -> None:
        self._tokenizer = tokenizer
        self._eos = eos_token_ids
        self._grammar = JsonGrammar(schema)
        self._bytes: dict[int, bytes] = {}

    def _token_bytes(self, token_id: int) -> bytes:
        found = self._bytes.get(token_id)
        if found is None:
            try:
                found = self._tokenizer.token_to_bytes(token_id)
            except (IndexError, KeyError):
                found = b""
            self._bytes[token_id] = found
        return found

    def allows(self, token_id: int) -> bool:
        if token_id in self._eos:
            return self._grammar.complete
        if self._grammar.complete:
            return False
        data = self._token_bytes(token_id)
        if not data or _SPECIAL.search(data):
            return False
        return self._grammar.clone().feed(data)

    def accept(self, token_id: int) -> None:
        if token_id not in self._eos:
            self._grammar.feed(self._token_bytes(token_id))
