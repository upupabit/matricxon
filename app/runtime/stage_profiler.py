"""Per-stage forward-pass timing for GET /api/live (pAIring's live workflow view).

Hooks attach only while someone watches (`LiveRegistry.watched`), so an unwatched model pays
nothing. `sync` must run on the model's worker thread: adding a hook while a forward pass iterates
the hook dict on another thread could fail.
"""

import time
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

_MATMUL_CLASSES = ("QuantizedLinear", "DevicePackedLinear", "Linear")


class StageProfiler:
    GROUPS = (
        "embed",
        "norm",
        "attention",
        "linear_attention",
        "ffn",
        "lm_head",
        "sampler",
        "matmul",
    )
    _ATTR = "_live_stage_profiler"

    def __init__(self, architecture: nn.Module) -> None:
        self._architecture = architecture
        self.totals: dict[str, float] = dict.fromkeys(self.GROUPS, 0.0)
        self.layer: int | None = None
        self.layer_count = len(self._layers()) or None
        self._handles: list[RemovableHandle] = []
        self._starts: dict[tuple[int, str], float] = {}
        # A tied output head (F.linear on token_embd) has no module: its time is the gap from the
        # last layer to the end of the forward pass (`forward_finished`).
        self._after_layers: float | None = None
        self._head_seen = False

    @classmethod
    def of(cls, architecture: nn.Module) -> "StageProfiler":
        """The architecture's own profiler, kept on it so it goes away with the model."""
        profiler = architecture.__dict__.get(cls._ATTR)
        if profiler is None:
            profiler = cls(architecture)
            architecture.__dict__[cls._ATTR] = profiler
        return profiler

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @property
    def times_stages(self) -> bool:
        """Stage times only on the CPU: GPU kernels run async, so wall time would be launch time."""
        device = getattr(self._architecture, "device", torch.device("cpu"))
        return device.type == "cpu"

    def sync(self, watched: bool) -> None:
        # Materializing replaces modules (e.g. lm_head), so hooks wait for real weights.
        if watched and not self.attached and getattr(self._architecture, "is_materialized", True):
            self._attach()
        elif not watched and self.attached:
            self._detach()

    def add(self, group: str, seconds: float) -> None:
        self.totals[group] += seconds

    def snapshot(self) -> dict[str, float]:
        return dict(self.totals)

    def forward_finished(self) -> None:
        """Called after each forward pass; times a tied output head."""
        if self._after_layers is not None and not self._head_seen and self.times_stages:
            self.totals["lm_head"] += time.perf_counter() - self._after_layers
        self._after_layers = None
        self._head_seen = False

    def _layers(self) -> list[nn.Module]:
        layers = getattr(self._architecture, "layers", None)
        return list(layers) if isinstance(layers, nn.ModuleList) else []

    def _attach(self) -> None:
        layers = self._layers()
        for index, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(self._layer_hook(index + 1)))
        if not self.times_stages:
            return
        if layers:
            self._handles.append(layers[-1].register_forward_hook(self._last_layer_done))
        self._visit(self._architecture)
        for module in self._architecture.modules():
            if type(module).__name__ in _MATMUL_CLASSES:
                self._time(module, "matmul")

    def _detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._starts.clear()
        self.layer = None
        self._after_layers = None

    def _visit(self, module: nn.Module) -> None:
        """Times the outermost module of each group; its children count toward it."""
        for child in module.children():
            group = self._group_of(child)
            if group is None:
                self._visit(child)
            else:
                self._time(child, group)

    def _group_of(self, module: nn.Module) -> str | None:
        if module is getattr(self._architecture, "lm_head", None):
            return "lm_head"
        if module is getattr(self._architecture, "token_embd", None):
            return "embed"
        name = type(module).__name__
        if "DeltaNet" in name or "Mamba" in name:
            return "linear_attention"
        if "Attention" in name:
            return "attention"
        if any(key in name for key in ("MLP", "FFN", "MoE")):
            return "ffn"
        if name in ("RMSNorm", "LayerNorm"):
            return "norm"
        return None

    def _time(self, module: nn.Module, group: str) -> None:
        key = (id(module), group)

        def before(_module: nn.Module, _args: tuple) -> None:
            self._starts[key] = time.perf_counter()

        def after(_module: nn.Module, _args: tuple, _output: object) -> None:
            started = self._starts.pop(key, None)
            if started is not None:
                self.totals[group] += time.perf_counter() - started
            if group == "lm_head":
                self._head_seen = True

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def _last_layer_done(self, _module: nn.Module, _args: tuple, _output: object) -> None:
        self._after_layers = time.perf_counter()

    def _layer_hook(self, number: int) -> Callable[[nn.Module, tuple], None]:
        def hook(_module: nn.Module, _args: tuple) -> None:
            self.layer = number

        return hook
