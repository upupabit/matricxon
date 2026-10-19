"""Gemma 4's small helpers: its causal/sliding mask, the weightless v_norm and GeGLU's gelu*up, each
using the fused CPU ops (app/native/fused_ops.py) when they apply."""

import torch
import torch.nn.functional as F

from app.native.fused_ops import FusedOps


def _causal_mask(
    q_len: int,
    kv_len: int,
    cache_offset: int,
    window: int | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Same causal rule as Mistral3DecoderLayer's `_causal_mask`, plus an optional sliding window:
    query i attends to
    key j iff `j <= i + cache_offset` (causal) and, if `window` is set, `j > i + cache_offset -
    window` (Gemma4's local layers).
    """
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    causal = kv_idx <= (q_idx + cache_offset)
    if window is None:
        return causal
    return causal & (kv_idx > (q_idx + cache_offset - window))


def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma4's `v_norm`: an active step with no learnable weight (`with_scale=False` in HF
    `Gemma4RMSNorm`; no
    `attn_v_norm.weight` in the GGUF), so a plain function rather than a module with nothing to
    load.
    """
    ops = FusedOps.active()
    if ops is not None:
        fused = ops.rms_norm(x, _ones(x.shape[-1]), eps)
        if fused is not None:
            return fused
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(input_dtype)


_ONES: dict[int, torch.Tensor] = {}


def _ones(dim: int) -> torch.Tensor:
    """A weight of ones for the fused RMSNorm (Gemma's v_norm has no weight)."""
    if dim not in _ONES:
        _ONES[dim] = torch.ones(dim, dtype=torch.float32)
    return _ONES[dim]


def gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """gelu_tanh(gate) * up: one fused pass at decode, torch otherwise."""
    ops = FusedOps.active()
    fused = ops.gelu_tanh_mul(gate, up) if ops is not None else None
    return fused if fused is not None else F.gelu(gate, approximate="tanh") * up
