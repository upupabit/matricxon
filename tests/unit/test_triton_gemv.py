"""The fused GPU kernels for packed Q8_0/Q4_K, run in Triton's CPU interpreter (no GPU needed)
against TorchDequantizer, plus when DevicePackedLinear uses them."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")

from app.architectures.triton_gemv import MAX_FUSED_TOKENS, TritonGemv  # noqa: E402
from app.gguf.constants import GGMLQuantizationType as T  # noqa: E402

_CHECK = textwrap.dedent(
    """
    import numpy as np
    import torch
    from app.architectures.triton_gemv import TritonGemv
    from app.gguf.constants import GGMLQuantizationType as T
    from app.gguf.dequant.torch_dequant import TorchDequantizer

    TritonGemv.enabled = True
    rng = np.random.default_rng(0)
    for t, block, size, nb in ((T.Q8_0, 32, 34, 8), (T.Q4_K, 256, 144, 2)):
        rows = 37  # not a multiple of the 16-row tile: the edge is masked
        raw = rng.integers(0, 256, size=(rows, nb * size), dtype=np.uint8)
        for b in range(nb):  # finite float16 scales (d, and dmin for Q4_K)
            for at in (0, 2) if t == T.Q4_K else (0,):
                scales = rng.uniform(0.001, 0.02, rows).astype(np.float16)
                raw[:, b * size + at : b * size + at + 2] = scales.view(np.uint8).reshape(rows, 2)
        packed = torch.from_numpy(raw)
        for tokens in (1, 3):
            x = torch.randn(1, tokens, block * nb)
            weight = TorchDequantizer.dequantize(packed.reshape(-1, size), t).reshape(rows, -1)
            expected = x @ weight.T
            got = TritonGemv.matmul(packed, t, x)
            assert got.shape == expected.shape
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-3), (t, tokens)
    print("ok")
    """
)


def test_the_kernels_match_the_torch_dequantizer_in_the_interpreter() -> None:
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(Path(__file__).parents[2]))
    result = subprocess.run(
        [sys.executable, "-c", _CHECK], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_only_enabled_supported_types_few_tokens_on_a_gpu(monkeypatch) -> None:
    x = torch.randn(1, 1, 256)
    monkeypatch.setattr(TritonGemv, "enabled", False)
    assert not TritonGemv.applies(T.Q4_K, x)
    monkeypatch.setattr(TritonGemv, "enabled", True)
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert not TritonGemv.applies(T.Q4_K, x)  # a CPU tensor, no interpreter
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    assert TritonGemv.applies(T.Q4_K, x) and TritonGemv.applies(T.Q8_0, x)
    assert not TritonGemv.applies(T.Q6_K, x)
    assert not TritonGemv.applies(T.Q4_K, torch.randn(1, MAX_FUSED_TOKENS + 1, 256))
