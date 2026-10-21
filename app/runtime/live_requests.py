"""Live progress of each chat reply, for GET /api/live (pAIring's live workflow view).

`LiveRequest` fields are plain attribute writes from the request and worker threads; snapshots
only read them, so no lock is needed per field. The registry's dict is guarded by a lock.
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from app.runtime.stage_profiler import StageProfiler
from app.schemas.live import LiveRequestEntry, LiveSnapshot


class LiveRequest:
    def __init__(self, request_id: str, model: str, clock: Callable[[], float] = time.monotonic):
        self.request_id = request_id
        self.model = model
        self._clock = clock
        self.phase = "loading"
        self.started = clock()
        self.load_seconds: float | None = None
        self.prompt_tokens = 0
        self.reused_tokens = 0
        self.prefilled_tokens = 0
        self.generated_tokens = 0
        self.prefill_started: float | None = None
        self.first_token_at: float | None = None
        self.finished_at: float | None = None
        self._token_times: deque[float] = deque(maxlen=16)
        self._profiler: StageProfiler | None = None
        self._profile_start: dict[str, float] = {}
        self._stage_seconds: dict[str, float] | None = None

    def loaded(self) -> None:
        self.load_seconds = self._clock() - self.started
        self.phase = "building"

    def queued(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.phase = "queued"

    def prefill(self, reused: int, profiler: StageProfiler) -> None:
        self.reused_tokens = reused
        self.prefill_started = self._clock()
        self._profiler = profiler
        self._profile_start = profiler.snapshot()
        self.phase = "prefill"

    def prefilled(self, tokens: int) -> None:
        self.prefilled_tokens = tokens

    def token(self) -> None:
        now = self._clock()
        if self.first_token_at is None:
            self.first_token_at = now
        self.generated_tokens += 1
        self._token_times.append(now)
        self.phase = "decode"

    def finish(self) -> None:
        if self.finished_at is not None:
            return
        self._stage_seconds = self.stage_seconds()
        self.finished_at = self._clock()
        self.phase = "done"

    def stage_seconds(self) -> dict[str, float]:
        if self._stage_seconds is not None:
            return self._stage_seconds
        if self._profiler is None:
            return {}
        now = self._profiler.snapshot()
        return {k: round(v - self._profile_start.get(k, 0.0), 4) for k, v in now.items()}

    def tokens_per_second(self) -> float | None:
        times = self._token_times
        if len(times) < 2 or times[-1] == times[0]:
            return None
        return (len(times) - 1) / (times[-1] - times[0])

    def first_token_seconds(self) -> float | None:
        if self.first_token_at is None or self.prefill_started is None:
            return None
        return self.first_token_at - self.prefill_started

    def entry(self) -> LiveRequestEntry:
        end = self.finished_at or self._clock()
        profiler = self._profiler
        running = self.finished_at is None and profiler is not None
        return LiveRequestEntry(
            request_id=self.request_id,
            model=self.model,
            phase=self.phase,
            elapsed_seconds=round(end - self.started, 3),
            load_seconds=self.load_seconds,
            prompt_tokens=self.prompt_tokens,
            reused_tokens=self.reused_tokens,
            prefilled_tokens=self.prefilled_tokens,
            generated_tokens=self.generated_tokens,
            first_token_seconds=self.first_token_seconds(),
            tokens_per_second=self.tokens_per_second(),
            layer=profiler.layer if running and profiler.attached else None,
            layer_count=profiler.layer_count if profiler else None,
            stage_seconds=self.stage_seconds(),
            stage_timing=bool(profiler and profiler.times_stages),
        )


class LiveRegistry:
    WATCH_SECONDS = 15.0  # timing stays on this long after the last GET /api/live
    KEEP_FINISHED_SECONDS = 30.0
    _shared: "LiveRegistry | None" = None

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._requests: dict[str, LiveRequest] = {}
        self._lock = threading.Lock()
        self._watched_until = 0.0

    @classmethod
    def shared(cls) -> "LiveRegistry":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @property
    def watched(self) -> bool:
        return self._clock() < self._watched_until

    def start(self, request_id: str | None, model: str) -> LiveRequest:
        live = LiveRequest(request_id or f"anon-{uuid.uuid4().hex[:12]}", model, self._clock)
        with self._lock:
            self._requests[live.request_id] = live
        return live

    def snapshot(self) -> LiveSnapshot:
        """Marks the registry watched and lists running replies plus recently finished ones."""
        now = self._clock()
        self._watched_until = now + self.WATCH_SECONDS
        with self._lock:
            for key, live in list(self._requests.items()):
                finished = live.finished_at
                if finished is not None and now - finished > self.KEEP_FINISHED_SECONDS:
                    del self._requests[key]
            requests = list(self._requests.values())
        return LiveSnapshot(requests=[live.entry() for live in requests])
