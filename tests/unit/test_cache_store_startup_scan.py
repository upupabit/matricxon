"""PersistentCacheStore's startup scan with a time budget (the server's start, app/dependencies.py):
it indexes saved chats before returning, and past the budget the rest continues on the writer thread
from where it stopped. Spills queue behind that rest."""

import threading
import time
from pathlib import Path

import pytest
import torch

from app import dependencies
from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_files import StoredEntry
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.kv_cache import KVCache
from app.runtime.prompt_slot import CacheSlot

KEY = bytes(range(32))
MODEL = "m|1"
CTX = 128


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _slot(token_ids: list[int]) -> CacheSlot:
    cache = KVCache([(2, 4)] * 3, CTX, torch.float32)
    for layer in range(3):
        cache.update(
            layer, torch.randn(1, 2, len(token_ids), 4), torch.randn(1, 2, len(token_ids), 4)
        )
    cache.advance(len(token_ids))
    return CacheSlot(cache, (CTX, torch.float32), list(token_ids), prompt_len=len(token_ids))


def _store(tmp_path: Path, budget: float | None) -> PersistentCacheStore:
    return PersistentCacheStore(
        tmp_path / "c",
        CacheCipher(KEY),
        PersistencePolicy(True, 100, 24),
        scan_budget_seconds=budget,
    )


def _saved_chats(tmp_path: Path, count: int) -> None:
    store = _store(tmp_path, None)
    for k in range(count):
        slot = _slot(list(range(100 * k + 1, 100 * k + 25)))
        slot.tag = f"chat-{k}"
        store.spill_async(MODEL, slot)
    store.flush()
    (tmp_path / "c" / "local-x.tmp").write_bytes(b"junk")


def test_without_a_budget_everything_is_indexed_before_construction_returns(
    tmp_path: Path,
) -> None:
    _saved_chats(tmp_path, 3)

    store = _store(tmp_path, None)

    assert len(store._index) == 3
    assert not (tmp_path / "c" / "local-x.tmp").exists()


def test_past_the_budget_the_rest_continues_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved_chats(tmp_path, 3)
    original = PersistentCacheStore._read_meta

    def slow_read(self: PersistentCacheStore, path: Path) -> StoredEntry:
        time.sleep(0.15)
        return original(self, path)

    monkeypatch.setattr(PersistentCacheStore, "_read_meta", slow_read)

    store = _store(tmp_path, 0.2)

    assert 1 <= len(store._index) < 3
    assert not (tmp_path / "c" / "local-x.tmp").exists()
    store.flush()
    assert len(store._index) == 3


def test_a_zero_budget_indexes_everything_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved_chats(tmp_path, 2)
    release = threading.Event()
    original = PersistentCacheStore._finish_startup_scan

    def held(self: PersistentCacheStore, started: float) -> None:
        release.wait(5)  # so the check below can't race the writer thread
        original(self, started)

    monkeypatch.setattr(PersistentCacheStore, "_finish_startup_scan", held)
    store = _store(tmp_path, 0.0)

    assert store._index == {}
    release.set()
    store.flush()
    assert len(store._index) == 2


def test_spills_wait_for_the_background_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    original = PersistentCacheStore._finish_startup_scan

    def slow_finish(self: PersistentCacheStore, started: float) -> None:
        release.wait(5)
        original(self, started)

    monkeypatch.setattr(PersistentCacheStore, "_finish_startup_scan", slow_finish)
    store = _store(tmp_path, 0.0)
    store.spill_async(MODEL, _slot(list(range(1, 25))))
    files = tmp_path / "c"

    assert not list(files.glob("*.mxc")) and not list(files.glob("*.tmp"))
    release.set()
    store.flush()
    assert len(list(files.glob("*.mxc"))) == 1


def test_budget_is_the_start_timeout_less_time_already_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MATRICXON_START_TIMEOUT counts from the process start (app.STARTED_AT); the scan stops 5 s
    before it, and a start that already used it all scans entirely in the background."""
    monkeypatch.setattr(dependencies.settings, "start_timeout", 90.0)
    monkeypatch.setattr(dependencies, "STARTED_AT", time.monotonic() - 20.0)
    assert 64.0 < dependencies._cache_scan_budget() <= 65.0

    monkeypatch.setattr(dependencies, "STARTED_AT", time.monotonic() - 200.0)
    assert dependencies._cache_scan_budget() == 0.0
