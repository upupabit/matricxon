"""Keeps a template's empty think block on earlier assistant turns, so a chat's next prompt starts
with exactly the text the model already read and wrote.

Qwen 3.5's template ends the generation prompt with `assistant\\n<think>\\n\\n</think>\\n\\n`
(thinking off), so the reply is generated after that block, but it renders earlier replies without
it. The prompt then stops matching the prompt cache where the last reply starts: an attention cache
cuts back cheaply, but a hybrid model's recurrent layers only roll back to a snapshot, so every turn
re-read the whole previous reply (41-77 tokens on Qwen3.5-4B, ~0.2 s each on the dev laptop).

Detected from the template itself (probe renders), not by model name; templates that keep their
generation prompt in history are left byte-identical. Turns that thought for real (thinking on) are
not covered: the template drops their reasoning, and the model saw it.
"""

from collections.abc import Callable

from jinja2 import TemplateError

from app.schemas.chat import ChatMessage
from app.server.errors import MatricxonError

# (messages, add_generation_prompt) -> rendered prompt, before any restoring.
RenderFn = Callable[[list[ChatMessage], bool], str]

_PROBE_USER = "Probe question one."
_PROBE_REPLY = "Probe reply one."
_PROBE_NEXT = "Probe question two."


class HistoryThinkBlock:
    def __init__(self, render: RenderFn) -> None:
        self._render = render
        self._detected = False
        self._header = ""
        self._block = ""

    @property
    def block(self) -> str:
        """The block put back after each assistant header; empty when the template needs none."""
        self._detect()
        return self._block

    def restore(self, prompt: str) -> str:
        """`prompt` with the block after every assistant header that lacks it (the generation prompt
        already has it)."""
        if not self.block:
            return prompt
        parts = prompt.split(self._header)
        restored = [parts[0]]
        for part in parts[1:]:
            restored.append(part if part.startswith(self._block) else self._block + part)
        return self._header.join(restored)

    def _detect(self) -> None:
        """Once, on first use: from_metadata runs per model on every GET /api/tags, so not there."""
        if self._detected:
            return
        self._detected = True
        try:
            user = [ChatMessage(role="user", content=_PROBE_USER)]
            history = [
                *user,
                ChatMessage(role="assistant", content=_PROBE_REPLY),
                ChatMessage(role="user", content=_PROBE_NEXT),
            ]
            before = self._render(user, False)
            generation = self._render(user, True)
            rendered = self._render(history, False)
        except (TemplateError, MatricxonError):
            return
        if not generation.startswith(before) or not rendered.startswith(before):
            return
        generation_tail = generation[len(before) :]
        history_tail = rendered[len(before) :]
        shared = 0
        while (
            shared < min(len(generation_tail), len(history_tail))
            and generation_tail[shared] == history_tail[shared]
        ):
            shared += 1
        header, block = generation_tail[:shared], generation_tail[shared:]
        # Only a dropped think block, right before the reply text: the case measured above.
        if header and "</think>" in block and history_tail[shared:].startswith(_PROBE_REPLY):
            self._header, self._block = header, block
