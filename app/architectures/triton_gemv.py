"""Fused dequantize-and-multiply GPU kernels (Triton) for packed Q8_0/Q4_K weights at decode
(`Settings.gpu_fused_kernels`, EXPERIMENTAL: checked in Triton's CPU interpreter against
`TorchDequantizer`, not yet run on a real GPU). They read the packed bytes once, with no dequantized
weight in VRAM, the way the CPU kernels do; `DevicePackedLinear` falls back to its torch path when
they don't apply. Same block layouts as `app/gguf/dequant/torch_dequant.py` (q8_0, q4_k)."""

import torch

from app.gguf.constants import GGMLQuantizationType as T

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs without Triton
    triton = None

# Above this many tokens (a prefill) the dequantize-then-matmul path is compute-bound and wins.
MAX_FUSED_TOKENS = 4
_ROWS_PER_PROGRAM = 16

if triton is not None:

    @triton.jit
    def _f16_at(ptr):
        """Little-endian float16 at `ptr` (one per row): two bytes combined, then bitcast."""
        lo = tl.load(ptr).to(tl.uint16)
        hi = tl.load(ptr + 1).to(tl.uint16)
        return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    @triton.jit
    def _q8_0_kernel(w, x, y, n_rows, n_blocks, row_bytes, ROWS: tl.constexpr):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        live = rows < n_rows
        lanes = tl.arange(0, 32)
        acc = tl.zeros((ROWS,), dtype=tl.float32)
        for b in range(n_blocks):
            base = w + rows * row_bytes + b * 34
            d = _f16_at(tl.where(live, base, w))
            q = tl.load(base[:, None] + 2 + lanes[None, :], mask=live[:, None], other=0)
            q = q.to(tl.int8, bitcast=True).to(tl.float32)
            xs = tl.load(x + b * 32 + lanes)
            acc += d * tl.sum(q * xs[None, :], axis=1)
        tl.store(y + rows, acc, mask=live)

    @triton.jit
    def _k4_scale_min(base, j: tl.constexpr):
        """ggml get_scale_min_k4 for sub-block j of each row's 12 scale bytes at `base`."""
        if j < 4:
            sc = tl.load(base + j).to(tl.int32) & 63
            m = tl.load(base + j + 4).to(tl.int32) & 63
        else:
            s_j4 = tl.load(base + j + 4).to(tl.int32)
            sc = (s_j4 & 0x0F) | ((tl.load(base + j - 4).to(tl.int32) >> 6) << 4)
            m = (s_j4 >> 4) | ((tl.load(base + j).to(tl.int32) >> 6) << 4)
        return sc.to(tl.float32), m.to(tl.float32)

    @triton.jit
    def _q4_k_kernel(w, x, y, n_rows, n_blocks, row_bytes, ROWS: tl.constexpr):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        live = rows < n_rows
        lanes = tl.arange(0, 32)
        acc = tl.zeros((ROWS,), dtype=tl.float32)
        for b in range(n_blocks):
            base = tl.where(live, w + rows * row_bytes + b * 144, w)
            d = _f16_at(base)
            dmin = _f16_at(base + 2)
            for g in tl.static_range(4):  # 32 bytes: low nibbles sub-block 2g, high 2g+1
                q = tl.load(base[:, None] + 16 + 32 * g + lanes[None, :]).to(tl.int32)
                x_lo = tl.load(x + b * 256 + 64 * g + lanes)
                x_hi = tl.load(x + b * 256 + 64 * g + 32 + lanes)
                sc_lo, m_lo = _k4_scale_min(base + 4, 2 * g)
                sc_hi, m_hi = _k4_scale_min(base + 4, 2 * g + 1)
                dot_lo = tl.sum((q & 0x0F).to(tl.float32) * x_lo[None, :], axis=1)
                dot_hi = tl.sum((q >> 4).to(tl.float32) * x_hi[None, :], axis=1)
                acc += d * (sc_lo * dot_lo + sc_hi * dot_hi)
                acc -= dmin * (m_lo * tl.sum(x_lo) + m_hi * tl.sum(x_hi))
        tl.store(y + rows, acc, mask=live)


_KERNELS = {T.Q8_0: ("_q8_0_kernel", 32, 34), T.Q4_K: ("_q4_k_kernel", 256, 144)}


class TritonGemv:
    """`y = x @ W.T` for packed rows `W` (`(out_features, row_bytes)` uint8 on the device)."""

    enabled = False  # Settings.gpu_fused_kernels, set at startup (app.main)

    @classmethod
    def applies(cls, ggml_type: int, x: torch.Tensor) -> bool:
        tokens = x.numel() // x.shape[-1]
        return (
            cls.enabled
            and triton is not None
            and ggml_type in _KERNELS
            and tokens <= MAX_FUSED_TOKENS
            and (x.is_cuda or _interpreting())
        )

    @staticmethod
    def matmul(packed: torch.Tensor, ggml_type: int, x: torch.Tensor) -> torch.Tensor:
        name, block_size, type_size = _KERNELS[ggml_type]
        out_features, in_features = packed.shape[0], x.shape[-1]
        rows = x.reshape(-1, in_features).float().contiguous()
        y = torch.empty((rows.shape[0], out_features), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(out_features, _ROWS_PER_PROGRAM),)
        kernel = globals()[name]
        for t in range(rows.shape[0]):
            kernel[grid](
                packed, rows[t], y[t], out_features, in_features // block_size,
                packed.shape[1], ROWS=_ROWS_PER_PROGRAM,
            )  # fmt: skip
        return y.reshape(*x.shape[:-1], out_features)


def _interpreting() -> bool:
    import os

    return os.environ.get("TRITON_INTERPRET") == "1"
