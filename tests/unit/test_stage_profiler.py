import torch
from torch import nn

from app.architectures.layers import RMSNorm
from app.runtime.stage_profiler import StageProfiler


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_norm = RMSNorm(4, 1e-6)
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.q_norm(x))


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ToyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(4, 1e-6)
        self.attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attn(self.norm(x)))


class ToyModel(nn.Module):
    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.token_embd = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(ToyDecoderLayer() for _ in range(layers))
        self.lm_head = nn.Linear(4, 8)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embd(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _run(model: ToyModel) -> None:
    with torch.no_grad():
        model(torch.tensor([[1, 2]]))


class TestStageProfiler:
    def test_times_each_group_and_tracks_the_layer_while_watched(self) -> None:
        model = ToyModel()
        profiler = StageProfiler.of(model)
        profiler.sync(watched=True)

        _run(model)

        for group in ("embed", "norm", "attention", "ffn", "lm_head", "matmul"):
            assert profiler.totals[group] > 0, group
        assert profiler.totals["linear_attention"] == 0
        assert profiler.layer == 3
        assert profiler.layer_count == 3

    def test_unwatched_removes_every_hook_and_stops_counting(self) -> None:
        model = ToyModel()
        profiler = StageProfiler.of(model)
        profiler.sync(watched=True)
        profiler.sync(watched=False)

        _run(model)

        assert not profiler.attached
        assert all(seconds == 0 for seconds in profiler.totals.values())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())

    def test_a_norm_inside_attention_counts_toward_attention_only(self) -> None:
        model = ToyModel(layers=1)
        profiler = StageProfiler.of(model)
        profiler.sync(watched=True)

        norm_hooks = model.layers[0].attn.q_norm._forward_pre_hooks

        assert not norm_hooks
        assert model.layers[0].norm._forward_pre_hooks

    def test_one_profiler_per_architecture(self) -> None:
        model = ToyModel()

        assert StageProfiler.of(model) is StageProfiler.of(model)


class TiedHeadModel(ToyModel):
    """Logits from the embedding matrix (F.linear), with no lm_head module."""

    def __init__(self) -> None:
        super().__init__()
        del self.lm_head

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embd(ids)
        for layer in self.layers:
            x = layer(x)
        return torch.nn.functional.linear(x, self.token_embd.weight)


class TestTiedHeadAndMaterializing:
    def test_a_tied_head_gets_the_time_after_the_last_layer(self) -> None:
        model = TiedHeadModel()
        profiler = StageProfiler.of(model)
        profiler.sync(watched=True)

        _run(model)
        profiler.forward_finished()

        assert profiler.totals["lm_head"] > 0

    def test_a_real_head_is_not_counted_twice(self) -> None:
        model = ToyModel()
        profiler = StageProfiler.of(model)
        profiler.sync(watched=True)
        _run(model)
        head = profiler.totals["lm_head"]

        profiler.forward_finished()

        assert profiler.totals["lm_head"] == head

    def test_waits_for_materialized_weights(self) -> None:
        model = ToyModel()
        model.is_materialized = False
        profiler = StageProfiler.of(model)

        profiler.sync(watched=True)
        assert not profiler.attached
        model.is_materialized = True
        profiler.sync(watched=True)
        assert profiler.attached
