"""estimate_ram_gb's per-file cache (app/models/ram_estimate.py), added because GET /api/tags
recomputed it (a GGUFReader parse) for every installed model on every request, making it
intermittently slow under load (the cache's docstring has the story). Asserted via a GGUFReader.read
call count, not estimated_ram_gb's values: a tiny fixture's rounded GB estimate is 0.0 whatever the
layer count, so an output assertion would pass even with caching broken."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.gguf.reader import GGUFReader
from app.models.ram_estimate import estimate_ram_gb
from app.routers.tags_router import TagsRequestHandler
from tests.tiny_gguf import build_tiny_mistral3_gguf


@pytest.fixture
def read_call_count(monkeypatch):
    real_read = GGUFReader.read
    counts = {"value": 0}

    def counting_read(self):
        counts["value"] += 1
        return real_read(self)

    monkeypatch.setattr(GGUFReader, "read", counting_read)
    return counts


def test_repeated_calls_for_the_same_unchanged_file_reuse_the_cached_result(
    tmp_path, read_call_count
):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")

    first = estimate_ram_gb(gguf_path)
    second = estimate_ram_gb(gguf_path)

    assert first == second
    assert read_call_count["value"] == 1


def test_a_different_quantized_native_enabled_triggers_its_own_real_read(tmp_path, read_call_count):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")

    estimate_ram_gb(gguf_path, quantized_native_enabled=False)
    estimate_ram_gb(gguf_path, quantized_native_enabled=True)

    assert read_call_count["value"] == 2


def test_a_file_replaced_at_the_same_path_invalidates_the_cache(tmp_path, read_call_count):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf", n_layer=1)
    estimate_ram_gb(gguf_path)

    # A re-pull to the same path rarely collides mtimes to the nanosecond on a fast filesystem;
    # nudged forward so the test exercises invalidation instead of flaking by coincidence.
    build_tiny_mistral3_gguf(gguf_path, n_layer=4)
    stat = Path(gguf_path).stat()
    os.utime(gguf_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    estimate_ram_gb(gguf_path)

    assert read_call_count["value"] == 2


def test_parallel_first_calls_share_one_read(tmp_path, read_call_count):
    gguf_path = build_tiny_mistral3_gguf(tmp_path / "model.gguf")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: estimate_ram_gb(gguf_path), range(8)))

    assert len(set(results)) == 1
    assert read_call_count["value"] == 1


def test_warm_up_fills_the_cache_in_the_background(monkeypatch) -> None:
    handled = threading.Event()
    monkeypatch.setattr(TagsRequestHandler, "handle", lambda _self: handled.set())

    TagsRequestHandler(catalog=None).warm_up()

    assert handled.wait(timeout=5)
