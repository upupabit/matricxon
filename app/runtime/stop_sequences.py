"""`options.stop`: ends a reply at the first stop text. The stop text itself is never sent, and text
that might be the start of one is held back until it is clear either way."""


class StopSequenceFilter:
    def __init__(self, stops: list[str] | None) -> None:
        self._stops = [s for s in (stops or []) if s]
        self._pending = ""
        self.stopped = False

    def push(self, text: str) -> str:
        """The text that is safe to send now (empty once stopped)."""
        if self.stopped or not self._stops:
            return "" if self.stopped else text
        self._pending += text
        hits = [i for i in (self._pending.find(s) for s in self._stops) if i >= 0]
        if hits:
            self.stopped = True
            out, self._pending = self._pending[: min(hits)], ""
            return out
        keep = self._partial_tail()
        out = self._pending[: len(self._pending) - keep]
        self._pending = self._pending[len(self._pending) - keep :]
        return out

    def finish(self) -> str:
        """Held-back text at the end of the reply (it never became a stop text)."""
        out, self._pending = ("" if self.stopped else self._pending), ""
        return out

    def _partial_tail(self) -> int:
        """Length of the longest end of the pending text that begins some stop text."""
        for size in range(min(len(self._pending), max(len(s) for s in self._stops) - 1), 0, -1):
            tail = self._pending[-size:]
            if any(s.startswith(tail) for s in self._stops):
                return size
        return 0
