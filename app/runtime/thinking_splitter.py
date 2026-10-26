"""Splits a streamed reply into reasoning and answer, like Ollama's `message.thinking`: text
inside `<think>...</think>` is reasoning, the rest the answer. A reply starts inside the block when
the prompt already opened it (Qwen 3.5 with `think: true` ends the generation prompt with
`<think>\n`); a model that opens one itself (`<think>` first, after whitespace) is handled too.
Partial tags split across chunks are held back until they resolve. Gemma 4 marks its reasoning
`<|channel>thought ... <channel|>` instead.
"""

# (open, close) reasoning markers: Qwen/DeepSeek style, then Gemma 4's thought channel.
MARKERS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))


class ThinkingSplitter:
    def __init__(self, inside_until: str | None = None) -> None:
        """`inside_until`: the closing marker if the prompt already opened the block."""
        self._state = "inside" if inside_until else "start"
        self._close = inside_until or MARKERS[0][1]
        self._pending = ""

    def push(self, text: str) -> tuple[str, str]:
        """(reasoning, answer) released by this chunk."""
        self._pending += text
        thinking, answer = "", ""
        while self._pending:
            if self._state == "start":
                stripped = self._pending.lstrip()
                opened = next(((o, c) for o, c in MARKERS if stripped.startswith(o)), None)
                if opened:
                    self._pending, self._state = stripped[len(opened[0]) :], "inside"
                    self._close = opened[1]
                elif any(o.startswith(stripped) for o, _ in MARKERS):  # a partial marker: wait
                    break
                else:
                    self._state = "answer"
            elif self._state == "inside":
                end = self._pending.find(self._close)
                if end == -1:
                    keep = self._partial_tail(self._pending, self._close)
                    thinking += self._pending[: len(self._pending) - keep]
                    self._pending = self._pending[len(self._pending) - keep :]
                    break
                thinking += self._pending[:end]
                self._pending = self._pending[end + len(self._close) :]
                self._state = "after"
            elif self._state == "after":  # the blank line the template puts after </think>
                stripped = self._pending.lstrip("\n")
                if not stripped:
                    break
                self._pending, self._state = stripped, "answer"
            else:
                answer += self._pending
                self._pending = ""
        return thinking, answer

    def finish(self) -> tuple[str, str]:
        """Whatever is still held back: unfinished reasoning stays reasoning."""
        rest, self._pending = self._pending, ""
        if self._state == "inside":
            return rest, ""
        return "", rest.lstrip("\n") if self._state == "after" else rest

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        """Length of the longest end of `text` that could start `tag`."""
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-size:]):
                return size
        return 0
