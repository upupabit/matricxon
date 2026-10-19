"""Reading a model's file back into the page cache ahead of use, in the background."""

import os
import time
from pathlib import Path

import pytest

from app.models.weight_prefetch import IDLE_SECONDS_BEFORE_PREFETCH, WeightPrefetch


def _wait_done(path: Path) -> None:
    deadline = time.monotonic() + 5
    while path in WeightPrefetch._running and time.monotonic() < deadline:
        time.sleep(0.01)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="Linux only")
def test_it_runs_once_per_file_at_a_time(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 4096)
    calls = []
    real = os.posix_fadvise

    def slow_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append(advice)
        time.sleep(0.2)
        real(fd, offset, length, advice)

    monkeypatch.setattr(os, "posix_fadvise", slow_fadvise)
    assert WeightPrefetch.start(path) is True
    assert WeightPrefetch.start(path) is False  # still running
    _wait_done(path)
    assert calls == [os.POSIX_FADV_WILLNEED]


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="Linux only")
def test_only_after_the_model_was_idle(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 4096)
    assert WeightPrefetch.after_idle(path, 1.0) is False
    assert WeightPrefetch.after_idle(path, IDLE_SECONDS_BEFORE_PREFETCH + 1) is True
    _wait_done(path)


def test_a_missing_file_is_only_logged(tmp_path: Path, caplog) -> None:
    path = tmp_path / "gone.gguf"
    WeightPrefetch._run(path)
    assert "couldn't prefetch" in caplog.text or not hasattr(os, "posix_fadvise")
