from pathlib import Path

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.registry import ArchitectureRegistry
from app.gguf.dequant.torch_dequant import TorchDequantizer
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.models.catalog import ModelCatalog
from app.models.handle import ModelHandle
from app.models.load_dtype import (
    QUANTIZED_NATIVE_WIRED_ARCHITECTURES,
    estimate_quantized_native_bytes,
    exact_bf16_bytes,
    plan_mixed_precision_load,
    select_load_dtype,
)
from app.models.load_settings import LoadSettings
from app.models.memory_guard import ensure_enough_memory_to_load
from app.models.tokenizer_dispatch import build_tokenizer
from app.models.worker import ModelWorker
from app.runtime.batch_decode import BatchDecoder
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_stop_tokens import extra_eos_token_ids
from app.runtime.chat_template import PromptBuilderFactory
from app.runtime.compute_device import ComputeDevice
from app.runtime.prompt_cache import PromptCache
from app.runtime.tool_call_formats import ToolCallFormats
from app.server.errors import DeviceUnavailableError


class HandleBuilder:
    """Loads one installed model into a ready ModelHandle (ModelManager decides when): memory
    check, load dtype, weights, tokenizer, worker and prompt cache."""

    def __init__(
        self,
        catalog: ModelCatalog,
        device: ComputeDevice,
        settings: LoadSettings,
        cache_store: PersistentCacheStore | None,
    ) -> None:
        self._catalog = catalog
        self._device = device
        self._settings = settings
        self._cache_store = cache_store
        self._registry = ArchitectureRegistry()

    def gguf_path(self, tag: str) -> Path:
        return self._catalog.gguf_path_for_tag(tag)

    def cache_budget_bytes(self, on_gpu: bool, free_before_load: int, weight_bytes: int) -> int:
        """Prompt-cache budget. On a GPU the cached KV lives in VRAM beside the weights, so it is
        capped at half the VRAM left once this model's weights are in (never above the setting; the
        weights may not be uploaded yet, hence the estimate)."""
        if not on_gpu:
            return self._settings.prompt_cache_budget_bytes
        vram_left = max(0, free_before_load - weight_bytes)
        return min(self._settings.prompt_cache_budget_bytes, vram_left // 2)

    def build(
        self, tag: str, keep_alive_seconds: int | None, now: float, loaded_tags: list[str]
    ) -> ModelHandle:
        installed = self._catalog.get(tag)
        gguf_path = self._catalog.gguf_path_for_tag(tag)

        # Parsed here (before the memory check), not left to GGUFModelLoader's constructor, so the
        # tensor shapes are available to compute bf16_bytes before deciding whether to attempt the
        # load. GGUFReader.read is a cheap KB-scale header parse, so GGUFModelLoader.__init__
        # re-parsing costs nothing worth avoiding. `.metadata` is kept to resolve the architecture
        # class below without another parse.
        parsed = GGUFReader(gguf_path).read()
        tensor_infos = parsed.tensor_infos
        bf16_bytes = exact_bf16_bytes(tensor_infos)
        architecture_cls = self._registry.resolve(parsed.metadata)

        # Why this breaker needs its own smaller estimate in this mode:
        # Settings.enable_quantized_native_compute / estimate_quantized_native_bytes.
        # `select_load_dtype` below still uses the full `bf16_bytes`: it decides float32-vs-bf16
        # only for tensors fully materialized either way (norms/embeddings/attn_q/attn_k), and a
        # smaller number could wrongly pick float32 with less headroom than assumed. The CPU
        # packed-weight kernels read the mmap, so a GPU model either materializes in bf16 or, with
        # gpu_weight_mode="packed", uploads supported quant types still packed.
        on_gpu = self._device.is_gpu
        if on_gpu and not architecture_cls.SUPPORTS_GPU:
            raise DeviceUnavailableError(
                f"architecture {architecture_cls.NAME!r} has no GPU support yet (CPU only)"
            )
        wired = architecture_cls.NAME in QUANTIZED_NATIVE_WIRED_ARCHITECTURES
        gpu_packed = on_gpu and wired and self._settings.gpu_weight_mode == "packed"
        quantized_native_enabled = gpu_packed or (
            self._settings.enable_quantized_native_compute and not on_gpu and wired
        )
        real_bytes_needed = estimate_quantized_native_bytes(
            tensor_infos,
            quantized_native_enabled,
            architecture_cls.NAME,
            packable=TorchDequantizer.supports if gpu_packed else None,
        )

        free_before_load = (self._device.free_memory_bytes() or 0) if on_gpu else 0
        ensure_enough_memory_to_load(
            tag,
            real_bytes_needed,
            self._settings.memory_safety_margin,
            loaded_tags,
            available_fn=self._device.free_memory_bytes if on_gpu else None,
            memory_kind="VRAM" if on_gpu else "memory",
        )
        # float32-vs-bf16 for what stays float: with quantized-native compute that is a small part
        # of the model, so size the decision from the real packed footprint, not the full unpacked
        # bf16 size, which made a 4B model always fall to bf16, ~10x slower here (no hardware bf16;
        # measured: Qwen3.5-4B decode 1.65 -> 0.18 tok/s).
        dtype = (
            self._device.load_dtype()
            if on_gpu
            else select_load_dtype(
                real_bytes_needed if quantized_native_enabled else bf16_bytes,
                self._settings.memory_safety_margin,
            )
        )

        # Mixed per-layer float32/bf16 precision: plan_mixed_precision_load (including why it
        # returns (None, dtype) by default).
        layer_dtypes, loader_dtype = plan_mixed_precision_load(
            tensor_infos,
            dtype,
            self._settings.memory_safety_margin,
            enabled=self._settings.enable_mixed_precision_loading and not on_gpu,
            supports_mixed_precision=architecture_cls is Mistral3TextArchitecture,
        )

        # No `with` block on purpose: on-the-fly dequant means `from_gguf` defers reading tensor
        # bytes until the first forward pass (ModelArchitecture._ensure_materialized), so the mmap
        # behind `loader` must stay open past this method; ownership passes to `architecture` once
        # from_gguf succeeds. If anything fails first, this closes it here instead of leaking it.
        loader = GGUFModelLoader(gguf_path, dtype=loader_dtype)
        try:
            # See Settings.enable_quantized_native_compute: only architectures in
            # QUANTIZED_NATIVE_WIRED_ARCHITECTURES accept this kwarg today.
            quantized_native_kwargs = (
                {"enable_quantized_native": True} if quantized_native_enabled else {}
            )
            if layer_dtypes is not None:
                architecture = architecture_cls.from_gguf(
                    loader, dtype=dtype, layer_dtypes=layer_dtypes, **quantized_native_kwargs
                )
            else:
                architecture = architecture_cls.from_gguf(
                    loader, dtype=dtype, **quantized_native_kwargs
                )
            tokenizer = build_tokenizer(loader.metadata)
            prompt_builder = PromptBuilderFactory.for_metadata(loader.metadata, tag=tag)
        except Exception:
            loader.close()
            raise
        try:
            architecture.place_on(self._device.torch_device, packed_weights=gpu_packed)
        except Exception:
            architecture.close()
            raise
        architecture.eval()
        batch = self._settings.max_decode_batch

        return ModelHandle(
            tag=tag,
            architecture=architecture,
            tokenizer=tokenizer,
            worker=ModelWorker(BatchDecoder(architecture, batch), batch),
            capabilities=installed.capabilities,
            size_bytes=installed.size_bytes,
            keep_alive_seconds=(
                keep_alive_seconds
                if keep_alive_seconds is not None
                else self._settings.default_keep_alive_seconds
            ),
            last_used_at=now,
            prompt_builder=prompt_builder,
            tool_call_format=ToolCallFormats.for_template(
                loader.metadata.get("tokenizer.chat_template"), loader.metadata.architecture
            ),
            extra_eos_token_ids=extra_eos_token_ids(loader.metadata),
            prompt_cache=PromptCache(
                self._settings.prompt_cache_slots,
                self.cache_budget_bytes(on_gpu, free_before_load, real_bytes_needed),
                ModelCacheTier(self._cache_store, f"{tag}|{installed.size_bytes}")
                if self._cache_store is not None
                else None,
                self._settings.prompt_cache_context_shift,
            ),
        )
