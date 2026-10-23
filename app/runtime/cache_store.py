import json
import logging
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.runtime.cache_crypto import CacheCipher, CacheDecryptError
from app.runtime.cache_files import MAGIC, SUFFIX, StoredEntry
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_reader import CacheReader
from app.runtime.cache_superseded import SupersededCopies
from app.runtime.cache_writer import CacheWriter
from app.runtime.slot_codec import SlotState

logger = logging.getLogger(__name__)

# Token granularity of prefix matching between a saved cache and a new prompt.
BLOCK_TOKENS = 64

__all__ = ["BLOCK_TOKENS", "MAGIC", "SUFFIX", "PersistentCacheStore", "StoredEntry"]


class PersistentCacheStore(CacheWriter, CacheReader):
    """Encrypted, size- and age-limited prompt-cache files on disk: the tier below the RAM pool.
    Writing is in cache_writer.py, restoring in cache_reader.py; this class owns the directory
    index, the limits and the admin actions.

    A slot is written when the pool evicts it, when its model unloads and on shutdown, never during
    a reply. File layout: `MAGIC | len | sealed meta | sealed body`, both AES-256-GCM sealed (body
    bound to meta), so nothing about a conversation (tokens, model, length) is readable without the
    key. The index is rebuilt at startup from the meta headers; a file failing authentication (wrong
    key, corruption) is deleted.

    A stored cache is consumed on restore (`take`): the RAM pool then owns it and writes it back
    when evicted. Limits come from the live `PersistencePolicy`: oldest files go first when over
    budget, and files older than the TTL are dropped.
    """

    # A lookup that finds nothing re-reads the shared dir if this long since the last read.
    MISS_RESCAN_SECONDS = 0.1

    def __init__(
        self,
        directory: Path,
        cipher: CacheCipher,
        policy: PersistencePolicy,
        node_id: str = "local",
        rescan_seconds: float = 5.0,
    ) -> None:
        self._dir = directory
        self._cipher = cipher
        self.policy = policy
        self._node_id = node_id
        self._rescan_seconds = rescan_seconds
        self._last_scan = 0.0
        # In a cluster only the owner of the cleanup lease deletes files by age or budget, so none
        # removes files the others count on. Alone, every node cleans.
        self.may_clean: Callable[[], bool] = lambda: True
        self._rejected: set[str] = set()
        self._lock = threading.Lock()
        self._index: dict[str, StoredEntry] = {}
        self._forgotten: set[str] = set()
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-spill")
        self._checkpoints: dict[tuple[str, str], SlotState] = {}  # newest copy waiting per chat
        self._spills_pending = 0  # chats leaving RAM, not yet on disk
        self._dir.mkdir(parents=True, exist_ok=True)
        self._scan()
        self.enforce_limits()

    # reading the directory

    def _scan(self) -> None:
        """Startup: drops this node's own leftovers (other nodes' are theirs) and indexes."""
        for path in self._dir.glob(f"{self._node_id}-*.tmp"):
            path.unlink(missing_ok=True)
        for path in self._dir.glob(f"*{self._claim_suffix}"):
            path.unlink(missing_ok=True)
        self._refresh(drop_unreadable=True)

    def _refresh(self, drop_unreadable: bool = False) -> None:
        """Syncs the index with the directory: files other nodes wrote or removed."""
        self._last_scan = time.monotonic()
        on_disk = {p.name: p for p in self._dir.glob(f"*{SUFFIX}")}
        with self._lock:
            known = set(self._index)
        for name in known - on_disk.keys():
            with self._lock:
                self._index.pop(name, None)
        for name, path in on_disk.items():
            if name in known or name in self._rejected:
                continue
            try:
                entry = self._read_meta(path)
            except FileNotFoundError:
                continue
            except (CacheDecryptError, OSError, ValueError, KeyError, TypeError):
                if drop_unreadable:
                    logger.warning("dropping unreadable prompt-cache file %s", name)
                    path.unlink(missing_ok=True)
                else:
                    self._rejected.add(name)
                continue
            with self._lock:
                self._index[name] = entry

    @property
    def _claim_suffix(self) -> str:
        return f".{self._node_id}.claim"

    def _rescan_if_due(self) -> None:
        if time.monotonic() - self._last_scan >= self._rescan_seconds:
            self._refresh()

    def _block_tokens(self) -> int:
        return BLOCK_TOKENS  # read at call time: tests shrink it

    def _read_meta(self, path: Path) -> StoredEntry:
        with path.open("rb") as handle:
            if handle.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a cache file")
            (meta_len,) = struct.unpack(">I", handle.read(4))
            meta = json.loads(self._cipher.open(handle.read(meta_len), MAGIC))
        return StoredEntry(**meta, size=path.stat().st_size)

    # limits and admin

    def _remove(self, name: str, path: Path | None = None) -> None:
        with self._lock:
            self._index.pop(name, None)
        (path or self._dir / name).unlink(missing_ok=True)

    def enforce_limits(self) -> None:
        """Drops files past the TTL, then the oldest until the total is within budget (only on the
        node owning the cleanup job)."""
        if not self.may_clean():
            return
        with self._lock:
            now = time.time()
            doomed = [
                n for n, e in self._index.items() if now - e.saved_at > self.policy.ttl_seconds
            ]
            keep = sorted(
                ((n, e) for n, e in self._index.items() if n not in doomed),
                key=lambda item: item[1].saved_at,
            )
            total = sum(e.size for _, e in keep)
            while keep and total > self.policy.budget_bytes:
                name, entry = keep.pop(0)
                doomed.append(name)
                total -= entry.size
        for name in doomed:
            self._remove(name)

    def forget(self, tag: str) -> int:
        """Deletes every stored file of the chat `tag` (and refuses its later spills); returns how
        many files went."""
        if not tag:
            return 0
        self._refresh()
        with self._lock:
            self._forgotten.add(tag)
            names = [n for n, e in self._index.items() if e.tag == tag]
        for name in names:
            self._remove(name)
        return len(names)

    def clear(self) -> None:
        self._refresh()
        with self._lock:
            names = list(self._index)
        for name in names:
            self._remove(name)

    def chat_count(self) -> int:
        """Distinct chats with a saved copy (one chat can have several files, e.g. branches)."""
        self._rescan_if_due()
        with self._lock:
            return SupersededCopies.chat_count(list(self._index.values()))

    def usage(self) -> tuple[int, int]:
        """(files, total bytes) currently stored."""
        self._rescan_if_due()
        with self._lock:
            return len(self._index), sum(e.size for e in self._index.values())
