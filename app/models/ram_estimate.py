import threading
from pathlib import Path

from app.gguf.reader import GGUFReader
from app.models.load_dtype import estimate_quantized_native_bytes


class RamEstimator:
    """The RAM a GGUF file needs to load, from a header-only parse, cached per file.

    Keyed by (path, mtime, size,...), not path, so a file replaced in place (a re-pull) invalidates
    automatically; one os.stat is cheap next to the parse it avoids. GET /api/tags asks for every
    installed model on every request. The parse runs under one lock, so parallel requests after a
    restart (cold cache, models on NFS: ~6 s for all of them) wait for one parse instead of each
    running their own and slowing each other past pAIring's timeouts.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int, bool, str], float] = {}
        self._lock = threading.Lock()

    def estimate_gb(
        self, gguf_path: str | Path, quantized_native_enabled: bool, architecture_name: str
    ) -> float:
        path = Path(gguf_path)
        stat = path.stat()
        key = (
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            quantized_native_enabled,
            architecture_name,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            tensor_infos = GGUFReader(path).read().tensor_infos
            real_bytes = estimate_quantized_native_bytes(
                tensor_infos, quantized_native_enabled, architecture_name
            )
            self._cache[key] = result = round(real_bytes / 1e9, 2)
        return result


_estimator = RamEstimator()


def estimate_ram_gb(
    gguf_path: str | Path,
    quantized_native_enabled: bool = False,
    architecture_name: str = "",
) -> float:
    """The RAM `gguf_path` needs to load: `estimate_quantized_native_bytes` of its tensor shapes in
    GB (equal to `exact_bf16_bytes` unless `quantized_native_enabled`), from a cheap header-only
    parse (GGUFReader.read).

    It does not apply `Settings.memory_safety_margin` (unlike
    `app.models.memory_guard.ensure_enough_memory_to_load`, the load-time gate): the padded figure
    would read as "this architecture needs more RAM than it does", though the padding is this app's
    load-time cushion.

    Shared by `/api/show` (one tag) and `/api/tags` (every tag, `tags_router.py`), so "how much RAM
    does Matricxon need" has one answer. `quantized_native_enabled` is passed by the caller from the
    live setting, so the number moves with the lower-RAM path.
    """
    return _estimator.estimate_gb(gguf_path, quantized_native_enabled, architecture_name)
