import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.handle_builder import HandleBuilder
from app.models.load_dtype import MEMORY_SAFETY_MARGIN
from app.models.load_settings import LoadSettings
from app.models.loaded_models import LoadedModels
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.compute_device import ComputeDevice
from app.runtime.memory_trim import memory_trimmer

# How many not-yet-started cancelled reply ids to remember (ModelManager.cancel_request).
_MAX_REMEMBERED_CANCELS = 1024


class ModelManager:
    """Loads GGUF models on demand and keeps up to `max_loaded` resident, evicting the
    least-recently-used one past its keep-alive.

    `max_loaded=2` matches pAIring's usage: its RAG path calls `embed` then `chat_stream` each turn,
    so 1 would reload a model every turn. There is no upper bound, but raising it on a
    RAM-constrained box relies on `_ensure_enough_memory_to_load`, which fails a load that clearly
    won't fit with `InsufficientMemoryError` instead of risking an OOM kill.

    Loads in bf16 when the hardware accelerates it (CUDA, or a CPU with `avx512_bf16`/`amx_bf16`),
    otherwise float32 if that demonstrably fits in available memory, else bf16
    (`app.models.load_dtype.select_load_dtype`): bf16 halves memory (~6.9GB vs ~13.7GB for a
    3.85B-param model) but is far slower on a CPU without native bf16, and float32 doesn't help if
    it thrashes swap.

    `clock` is injectable so eviction/expiry logic is testable with a fake clock. Loading one
    model is HandleBuilder's job; keeping and evicting them is LoadedModels'.
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        max_loaded: int = 2,
        default_keep_alive_seconds: int = 300,
        memory_safety_margin: float = MEMORY_SAFETY_MARGIN,
        enable_mixed_precision_loading: bool = False,
        enable_quantized_native_compute: bool = False,
        clock: Callable[[], float] = time.monotonic,
        prompt_cache_slots: int = 4,
        prompt_cache_context_shift: bool = True,
        prompt_cache_budget_mb: int = 2048,
        max_decode_batch: int = 8,
        device: str = "cpu",
        gpu_weight_mode: str = "dequantized",
        cache_store: PersistentCacheStore | None = None,
    ) -> None:
        # Raises DeviceUnavailableError for a missing GPU (never a silent CPU fallback).
        self._device = ComputeDevice.resolve(device)
        self._cache_store = cache_store
        self.settings = LoadSettings(
            max_loaded=max_loaded,
            default_keep_alive_seconds=default_keep_alive_seconds,
            memory_safety_margin=memory_safety_margin,
            enable_mixed_precision_loading=enable_mixed_precision_loading,
            enable_quantized_native_compute=enable_quantized_native_compute,
            prompt_cache_slots=prompt_cache_slots,
            prompt_cache_context_shift=prompt_cache_context_shift,
            prompt_cache_budget_bytes=prompt_cache_budget_mb << 20,
            max_decode_batch=max_decode_batch,
            gpu_weight_mode=gpu_weight_mode,
        )
        self._builder = HandleBuilder(catalog, self._device, self.settings, cache_store)
        self._clock = clock
        self._loaded = LoadedModels()
        # Tags stop-requested while they had no handle yet (unload says why that is common, not a
        # stray call): on the slow-CPU target loading takes tens of seconds, and a caller can ask to
        # stop well before it finishes.
        self._cancelled_loads: set[str] = set()
        self._cancelled_requests: OrderedDict[str, None] = OrderedDict()
        # One lock per tag: requests for a model that is still loading (chats, a cluster warm-up)
        # wait for that load instead of loading a second copy.
        self._load_locks: dict[str, threading.Lock] = {}
        self._load_locks_guard = threading.Lock()

    def get_or_load(self, tag: str, keep_alive_seconds: int | None = None) -> ModelHandle:
        with self._load_locks_guard:
            lock = self._load_locks.setdefault(tag, threading.Lock())
        with lock:
            return self._get_or_load(tag, keep_alive_seconds)

    def _get_or_load(self, tag: str, keep_alive_seconds: int | None) -> ModelHandle:
        now = self._clock()
        self._loaded.evict_expired(now)

        handle = self._loaded.get(tag)
        if handle is not None:
            handle.touch(now, keep_alive_seconds)
            return handle

        if len(self._loaded) >= self.settings.max_loaded:
            self._loaded.evict_least_recently_used(self.settings.max_loaded)

        # A fresh load starts clean: a leftover cancellation flag belongs to a previous finished
        # attempt for this tag.
        self._cancelled_loads.discard(tag)
        handle = self._load(tag, keep_alive_seconds, now)
        handle.worker.on_job_done = lambda: handle.touch(self._clock())
        self._loaded.add(handle)
        # Loading leaves large temporary buffers behind.
        memory_trimmer.schedule()
        return handle

    def cancel_request(self, tag: str, request_id: str) -> None:
        """Stops one reply (`ChatRequest.request_id`) without unloading anything: the model, other
        users' replies and caches are untouched. If the reply isn't on a worker yet (still loading
        the model, or between router and worker) the id is remembered, so `pop_cancelled_request`
        can drop it when it would start. See `ModelWorker.cancel_job`."""
        handle = self._loaded.get(tag)
        if handle is not None and handle.worker.cancel_job(request_id):
            return
        self._cancelled_requests[request_id] = None
        while len(self._cancelled_requests) > _MAX_REMEMBERED_CANCELS:
            self._cancelled_requests.popitem(last=False)

    def pop_cancelled_request(self, request_id: str) -> bool:
        """True (and consumes the note) if `cancel_request` ran for this id before it started."""
        return self._cancelled_requests.pop(request_id, 0) is None

    def pop_cancelled_load(self, tag: str) -> bool:
        """True (and consumes the flag) if unload was called for `tag` while it had no handle,
        i.e. while a get_or_load for it was in `_load`'s many-seconds dequantization step,
        before any ModelWorker existed for unload's request_stop to reach. The caller
        (ChatRequestHandler.handle, right after get_or_load returns) uses it to answer "done,
        nothing generated" instead of starting a generation nobody wants; the loaded handle stays
        cached (the work was done)."""
        if tag in self._cancelled_loads:
            self._cancelled_loads.discard(tag)
            return True
        return False

    def unload(self, tag: str) -> None:
        """Evicts `tag`'s handle and, if it has one, tells its worker to abort the running job
        (`ModelWorker.request_stop`). Popping the dict entry alone (the old behavior) left a
        mid-generation worker thread running: it kept computing every token up to
        max_new_tokens/EOS, invisible to `/api/ps` and still using the CPU. This is the "stop"
        contract of ChatRequest.is_unload_call (sent by app.services.matricxon_client.stop_model,
        e.g. after a mid-stream message delete).

        `wait_until_idle` runs before `architecture.close`: `request_stop` only sets a flag seen at
        the next per-layer/per-token check, so `close` could run mid-`forward` while
        quantized-native compute read a `QuantizedLinear`'s raw-mmap view through a fused GEMV
        kernel, crashing with `BufferError: cannot close exported pointers exist`. Waiting first
        means `close` runs after the worker has stopped touching the weights.

        If `tag` has no handle, a get_or_load for it is most likely still loading
        (`pop_cancelled_load`), the common case since loading can outlast a user's patience.
        Recording the stop lets that load notice, when it finishes, that nobody wants the
        generation.
        """
        handle = self._loaded.pop(tag)
        if handle is not None:
            handle.worker.request_stop()
            handle.worker.wait_until_idle()
            LoadedModels.retire(handle, "unload request")
        else:
            self._cancelled_loads.add(tag)

    def holds_chat(self, tag: str) -> bool:
        """Is a conversation of the chat `tag` in RAM on any loaded model? (cluster routing)"""
        return any(handle.prompt_cache.holds(tag) for handle in self.list_loaded())

    def spill_chat(self, tag: str) -> int:
        """Writes the chat's idle RAM conversations to the shared disk tier and waits for the files,
        so another node can continue the chat; the models stay loaded. Returns how many."""
        moved = sum(handle.prompt_cache.spill_chat(tag) for handle in self.list_loaded())
        if moved and self._cache_store is not None:
            self._cache_store.flush()
        return moved

    def shutdown(self) -> None:
        """Server stop: halts every reply, saves every model's cached conversations to disk and
        waits for the writes, so a restart keeps long chats cheap to resume."""
        for tag in self._loaded.tags():
            self.unload(tag)
        if self._cache_store is not None:
            self._cache_store.flush()

    def set_limits(
        self,
        *,
        max_loaded: int | None = None,
        default_keep_alive_seconds: int | None = None,
        memory_safety_margin: float | None = None,
        prompt_cache_slots: int | None = None,
        prompt_cache_budget_mb: int | None = None,
        max_decode_batch: int | None = None,
        enable_quantized_native_compute: bool | None = None,
        gpu_weight_mode: str | None = None,
    ) -> None:
        """New values without a restart; None keeps one. Loaded models take the keep-alive, cache
        and concurrency limits now; the other values apply from the next load."""
        new = {
            "max_loaded": max_loaded,
            "default_keep_alive_seconds": default_keep_alive_seconds,
            "memory_safety_margin": memory_safety_margin,
            "prompt_cache_slots": prompt_cache_slots,
            "prompt_cache_budget_bytes": (
                prompt_cache_budget_mb << 20 if prompt_cache_budget_mb is not None else None
            ),
            "max_decode_batch": max_decode_batch,
            "enable_quantized_native_compute": enable_quantized_native_compute,
            "gpu_weight_mode": gpu_weight_mode,
        }
        for name, value in new.items():
            if value is not None:
                setattr(self.settings, name, value)
        cache_changed = prompt_cache_slots is not None or prompt_cache_budget_mb is not None
        for handle in self._loaded.handles():
            if default_keep_alive_seconds is not None:
                handle.keep_alive_seconds = default_keep_alive_seconds
            if cache_changed:
                budget = self.settings.prompt_cache_budget_bytes
                if self.on_gpu:  # never above the VRAM cap set at load
                    budget = min(budget, handle.prompt_cache.budget_bytes)
                handle.prompt_cache.set_limits(self.settings.prompt_cache_slots, budget)
            if max_decode_batch is not None:
                handle.worker.set_max_active(max_decode_batch)

    @property
    def on_gpu(self) -> bool:
        return self._device.is_gpu

    def now(self) -> float:
        """Exposes the manager's own (possibly fake) clock, so a caller computing a `ModelHandle`'s
        `expires_in` reads the same clock its `last_used_at` came from (a wall-clock
        `time.monotonic` at the API boundary would disagree under a fake clock)."""
        return self._clock()

    def list_loaded(self) -> list[ModelHandle]:
        self._loaded.evict_expired(self._clock())
        return self._loaded.handles()

    def evict_expired(self) -> None:
        self._loaded.evict_expired(self._clock())

    def _load(self, tag: str, keep_alive_seconds: int | None, now: float) -> ModelHandle:
        """One model loaded (HandleBuilder.build); a seam tests replace to slow or count loads."""
        return self._builder.build(tag, keep_alive_seconds, now, self._loaded.tags())
