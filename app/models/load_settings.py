from dataclasses import dataclass


@dataclass
class LoadSettings:
    """ModelManager's tunable values, shared with HandleBuilder and changed live by
    `ModelManager.set_limits` (Settings explains each one)."""

    max_loaded: int
    default_keep_alive_seconds: int
    memory_safety_margin: float
    # Off by default (Settings.enable_mixed_precision_loading: measured slower on the target).
    enable_mixed_precision_loading: bool
    # Off by default (Settings.enable_quantized_native_compute): a permanent user choice.
    enable_quantized_native_compute: bool
    prompt_cache_slots: int
    prompt_cache_context_shift: bool
    prompt_cache_budget_bytes: int
    max_decode_batch: int
    gpu_weight_mode: str
