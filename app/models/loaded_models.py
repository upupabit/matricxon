import logging

from app.models.handle import ModelHandle
from app.runtime.memory_trim import memory_trimmer

logger = logging.getLogger(__name__)


class LoadedModels:
    """The models in memory, by tag, and the three ways one leaves: its keep-alive ran out, an
    unload request, or making room for another. Each is logged with its reason.

    A handle whose worker is_busy (ModelWorker.is_busy says why) is never evicted: the timer
    running out doesn't make it safe to yank the handle from a job still using it; the worker would
    keep computing, orphaned from this dict and invisible to /api/ps, with unload (the one way to
    stop it) unable to find it.
    """

    def __init__(self) -> None:
        self._handles: dict[str, ModelHandle] = {}

    def __len__(self) -> int:
        return len(self._handles)

    def get(self, tag: str) -> ModelHandle | None:
        return self._handles.get(tag)

    def add(self, handle: ModelHandle) -> None:
        self._handles[handle.tag] = handle

    def tags(self) -> list[str]:
        return list(self._handles)

    def handles(self) -> list[ModelHandle]:
        return list(self._handles.values())

    def pop(self, tag: str) -> ModelHandle | None:
        return self._handles.pop(tag, None)

    def evict_expired(self, now: float) -> None:
        for tag in [
            tag
            for tag, handle in self._handles.items()
            if handle.is_expired(now) and not handle.worker.is_busy()
        ]:
            handle = self._handles.pop(tag)
            idle_min, keep_min = (now - handle.last_used_at) / 60, handle.keep_alive_seconds // 60
            self.retire(handle, f"idle {idle_min:.0f} min, keep-alive {keep_min} min")

    def evict_least_recently_used(self, max_loaded: int) -> None:
        # If every loaded handle is busy this is a deliberate no-op, and get_or_load's caller loads
        # the new model alongside, temporarily over max_loaded rather than corrupting an in-flight
        # generation. A soft cap under contention is the honest tradeoff.
        idle = {tag: handle for tag, handle in self._handles.items() if not handle.worker.is_busy()}
        if not idle:
            return
        lru_tag = min(idle, key=lambda tag: idle[tag].last_used_at)
        self.retire(self._handles.pop(lru_tag), f"making room (at most {max_loaded} loaded)")

    @staticmethod
    def retire(handle: ModelHandle, reason: str) -> None:
        """Drops a handle for good: its idle cached conversations go to the disk tier first (a
        background write; the slots are independent of the weights closed right after)."""
        logger.info("unloading %s: %s", handle.tag, reason)
        handle.prompt_cache.persist_all()
        handle.worker.shutdown()
        handle.architecture.close()
        memory_trimmer.schedule()
