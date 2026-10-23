"""A chat still in RAM is copied to disk after a reply (`save_every_seconds`), so a crash or a
forced stop keeps it, and an unload doesn't write the same state twice."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.runtime import cache_store, cache_tier, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=96)
TURN1 = [3 + (i * 7) % 200 for i in range(30)]


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


@pytest.fixture
def llama(tmp_path: Path) -> Iterator[LlamaArchitecture]:
    model = LlamaArchitecture.from_gguf(
        GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "l.gguf"), dtype=torch.float32)
    )
    yield model
    model.close()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(cache_tier.time, "monotonic", clock)
    return clock


def _store(tmp_path: Path, save_every: int = 30) -> PersistentCacheStore:
    policy = PersistencePolicy(True, 100, 24, save_every_seconds=save_every)
    return PersistentCacheStore(tmp_path / "cache", CacheCipher(bytes(range(32))), policy)


def _reply(model: object, pool: PromptCache, ids: list[int], tag: str = "chat-1") -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool, prefill_chunk=8)
    request = GenerationRequest(torch.tensor([ids]), GREEDY, cache_tag=tag)
    return engine.generate(request).token_ids


def test_a_reply_is_saved_while_the_chat_stays_in_ram(
    llama: LlamaArchitecture, tmp_path: Path, clock: Clock
) -> None:
    store = _store(tmp_path)
    pool = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    reply = _reply(llama, pool, TURN1)
    store.flush()

    assert store.usage()[0] == 1 and pool.holds("chat-1")
    # A forced stop: nothing else is written. A new process still finds the chat.
    after = PromptCache(4, tier=ModelCacheTier(_store(tmp_path), "m|1"))
    _reply(llama, after, TURN1 + reply + [9, 8])
    assert after.reused_tokens >= 24


def test_saves_wait_for_the_interval(
    llama: LlamaArchitecture, tmp_path: Path, clock: Clock
) -> None:
    store = _store(tmp_path, save_every=30)
    pool = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    reply = _reply(llama, pool, TURN1)
    store.flush()
    first = store.usage()[1]

    clock.now += 10
    turn2 = TURN1 + reply + [9, 8]
    reply2 = _reply(llama, pool, turn2)
    store.flush()
    assert store.usage()[1] == first  # too soon: the first copy stays

    clock.now += 30
    _reply(llama, pool, turn2 + reply2 + [7])
    store.flush()
    assert store.usage()[0] == 1 and store.usage()[1] > first  # the newer copy replaced it


def test_zero_saves_only_when_the_chat_leaves_ram(
    llama: LlamaArchitecture, tmp_path: Path, clock: Clock
) -> None:
    store = _store(tmp_path, save_every=0)
    pool = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    _reply(llama, pool, TURN1)
    store.flush()
    assert store.usage()[0] == 0

    pool.persist_all()
    store.flush()
    assert store.usage()[0] == 1


def test_an_unload_skips_a_chat_already_saved(
    llama: LlamaArchitecture, tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    pool = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    _reply(llama, pool, TURN1)
    store.flush()
    spills: list[object] = []
    monkeypatch.setattr(store, "spill_async", lambda model_id, slot: spills.append(slot))

    pool.persist_all()

    assert spills == []
