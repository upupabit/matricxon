import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.slot_codec import SlotState

MAGIC = b"MXC1"
SUFFIX = ".mxc"


@dataclass
class StoredEntry:
    """What the store knows about a file without reading its body (sealed in its header)."""

    model_id: str
    num_ctx: int
    dtype_name: str
    length: int
    blocks: list[str]
    saved_at: float
    tag: str = ""
    size: int = 0

    def meta_json(self) -> bytes:
        return json.dumps({k: v for k, v in asdict(self).items() if k != "size"}).encode()


class CacheFilesBase:
    """The state and helpers `PersistentCacheStore` (cache_store.py) shares with its writing
    (cache_writer.py) and restoring (cache_reader.py) halves. The store sets every attribute and
    implements every method below."""

    MISS_RESCAN_SECONDS: float
    policy: PersistencePolicy
    _dir: Path
    _cipher: CacheCipher
    _node_id: str
    _lock: threading.Lock
    _index: dict[str, StoredEntry]
    _forgotten: set[str]
    _writer: ThreadPoolExecutor
    _checkpoints: dict[tuple[str, str], SlotState]
    _spills_pending: int
    _last_scan: float

    def _block_tokens(self) -> int:
        raise NotImplementedError

    @property
    def _claim_suffix(self) -> str:
        raise NotImplementedError

    def _refresh(self, drop_unreadable: bool = False) -> None:
        raise NotImplementedError

    def _rescan_if_due(self) -> None:
        raise NotImplementedError

    def _remove(self, name: str, path: Path | None = None) -> None:
        raise NotImplementedError

    def enforce_limits(self) -> None:
        raise NotImplementedError
