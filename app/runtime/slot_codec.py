import io
from dataclasses import dataclass

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.mamba_cache import HybridSnapshot, NemotronHHybridCache
from app.runtime.prompt_slot import CacheSlot

# Tokens below this are cheaper to recompute than to write and read back.
MIN_PERSIST_TOKENS = 64


@dataclass
class EncodedSlot:
    """A slot ready to be sealed: what the store indexes it by, plus the serialized tensors."""

    token_ids: list[int]
    num_ctx: int
    dtype_name: str
    payload: bytes


@dataclass
class SlotState:
    """A slot's saved part, copied out (`SlotCodec.capture`) and not yet serialized."""

    tensors: dict
    num_ctx: int
    dtype_name: str


class SlotCodec:
    """Turns a `CacheSlot` into bytes and back. Only what a later request can reuse is written: a
    plain cache's whole KV; a hybrid cache's KV up to its newest recurrent snapshot (past that
    nothing can be resumed) plus the snapshots, never the live recurrent state, which a snapshot
    always replaces before a slot is used again (`CacheSlot.rewind`/`fork`)."""

    @staticmethod
    def encode(slot: CacheSlot) -> EncodedSlot | None:
        """None when the slot isn't worth persisting (too short, or a hybrid without snapshots)."""
        state = SlotCodec.capture(slot)
        return None if state is None else SlotCodec.serialize(state)

    @staticmethod
    def capture(slot: CacheSlot) -> SlotState | None:
        """A copy of what `encode` writes, independent of the slot from then on (the KV is cloned),
        so it can be serialized on another thread while the slot serves the next reply."""
        keep = max(slot.snapshots, default=0) if slot.hybrid else len(slot.token_ids)
        if keep < MIN_PERSIST_TOKENS:
            return None
        keys, values = slot.cache.export(keep)
        num_ctx, dtype = slot.key
        state = {
            "token_ids": slot.token_ids[:keep],
            "prompt_len": min(slot.prompt_len, keep),
            "keys": keys,
            "values": values,
            "snapshots": [
                {"length": s.length, "conv": s.conv_state, "ssm": s.ssm_state}
                for s in slot.snapshots.values()
            ],
        }
        return SlotState(state, num_ctx, str(dtype))

    @staticmethod
    def serialize(state: SlotState) -> EncodedSlot:
        buffer = io.BytesIO()
        torch.save(state.tensors, buffer)
        return EncodedSlot(
            state.tensors["token_ids"], state.num_ctx, state.dtype_name, buffer.getvalue()
        )

    @staticmethod
    def decode(
        payload: bytes, architecture: ModelArchitecture, num_ctx: int, dtype: torch.dtype
    ) -> CacheSlot:
        """Rebuilds the slot on a fresh cache from `architecture`. Raises `ValueError` if it doesn't
        fit this model (a stale save); the caller treats it as a miss."""
        state = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        snapshots = {
            # Converted like the KV in `cache.load`: the model may have loaded in another precision.
            s["length"]: HybridSnapshot(
                s["length"], [t.to(dtype) for t in s["conv"]], [t.to(dtype) for t in s["ssm"]]
            )
            for s in state["snapshots"]
        }
        if bool(snapshots) != isinstance(cache, NemotronHHybridCache):
            raise ValueError("saved cache kind differs from this model's")
        cache.load(state["keys"], state["values"])
        return CacheSlot(
            cache, (num_ctx, dtype), state["token_ids"], snapshots, prompt_len=state["prompt_len"]
        )
