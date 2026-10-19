"""Asks the OS to read a model's GGUF file back into the page cache ahead of use. Weights are read
through a memory map, so under memory pressure the OS drops them, and the next request then faults
them in a page at a time: on the dev machine (weights on an NFS mount) a 6-token prompt took 49 s
cold vs 1.2 s warm. One `posix_fadvise(WILLNEED)` reads the file in large chunks instead, in the
background, so the request doesn't wait for it to finish."""

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# A model unused this long may have lost its pages; prefetch it when the next request comes.
IDLE_SECONDS_BEFORE_PREFETCH = 60.0


class WeightPrefetch:
    _running: set[Path] = set()
    _lock = threading.Lock()

    @classmethod
    def start(cls, path: Path) -> bool:
        """Starts reading `path` ahead in a background thread; False when unsupported (not Linux)
        or already running for this file."""
        if not hasattr(os, "posix_fadvise"):
            return False
        with cls._lock:
            if path in cls._running:
                return False
            cls._running.add(path)
        threading.Thread(target=cls._run, args=(path,), name="weight-prefetch", daemon=True).start()
        return True

    @classmethod
    def after_idle(cls, path: Path, idle_seconds: float) -> bool:
        """Starts it when the model was unused long enough that its pages may be gone."""
        return idle_seconds > IDLE_SECONDS_BEFORE_PREFETCH and cls.start(path)

    @classmethod
    def _run(cls, path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            finally:
                os.close(fd)
            logger.info("prefetching weights of %s", path.name)
        except OSError as exc:
            logger.warning("couldn't prefetch %s: %s", path.name, exc)
        finally:
            with cls._lock:
                cls._running.discard(path)
