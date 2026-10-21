import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import live_router
from app.runtime.chat_engine import ChatEngine
from app.runtime.live_requests import LiveRegistry, LiveRequest
from app.runtime.stage_profiler import StageProfiler
from tests.unit.test_chat_engine import _AlwaysTokenOneDecoder, _request
from tests.unit.test_stage_profiler import ToyModel


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestLiveRequest:
    def test_walks_through_the_phases_with_timings(self) -> None:
        clock = FakeClock()
        live = LiveRequest("r1", "m", clock)
        clock.now += 2.0
        live.loaded()
        live.queued(prompt_tokens=50)
        live.prefill(reused=20, profiler=StageProfiler.of(ToyModel()))
        live.prefilled(30)
        clock.now += 1.5
        live.token()
        for _ in range(4):
            clock.now += 0.25
            live.token()

        entry = live.entry()

        assert entry.phase == "decode"
        assert entry.load_seconds == 2.0
        assert (entry.prompt_tokens, entry.reused_tokens, entry.prefilled_tokens) == (50, 20, 30)
        assert entry.first_token_seconds == 1.5
        assert entry.generated_tokens == 5
        assert entry.tokens_per_second == 4.0

    def test_finish_freezes_elapsed_and_stage_times(self) -> None:
        clock = FakeClock()
        live = LiveRequest("r1", "m", clock)
        profiler = StageProfiler.of(ToyModel())
        live.prefill(reused=0, profiler=profiler)
        profiler.add("ffn", 0.5)
        clock.now += 3.0
        live.finish()
        profiler.add("ffn", 9.0)
        clock.now += 10.0

        entry = live.entry()

        assert entry.phase == "done"
        assert entry.elapsed_seconds == 3.0
        assert entry.stage_seconds["ffn"] == 0.5


class TestLiveRegistry:
    def test_snapshot_marks_watched_for_a_while(self) -> None:
        clock = FakeClock()
        registry = LiveRegistry(clock)
        assert not registry.watched

        registry.snapshot()
        assert registry.watched
        clock.now += LiveRegistry.WATCH_SECONDS + 1
        assert not registry.watched

    def test_drops_finished_requests_after_the_keep_window(self) -> None:
        clock = FakeClock()
        registry = LiveRegistry(clock)
        registry.start("old", "m").finish()
        clock.now += LiveRegistry.KEEP_FINISHED_SECONDS + 1
        registry.start("new", "m")

        ids = [entry.request_id for entry in registry.snapshot().requests]

        assert ids == ["new"]

    def test_old_entries_are_dropped_even_when_no_one_watches(self) -> None:
        clock = FakeClock()
        registry = LiveRegistry(clock)
        registry.start("old", "m").finish()
        clock.now += LiveRegistry.KEEP_FINISHED_SECONDS + 1

        registry.start("new", "m")

        assert list(registry._requests) == ["new"]
        assert not registry.watched

    def test_a_finished_request_lets_go_of_the_model(self) -> None:
        live = LiveRequest("r1", "m", FakeClock())
        model = ToyModel()
        live.prefill(reused=0, profiler=StageProfiler.of(model))
        live.finish()

        entry = live.entry()

        assert live._profiler is None
        assert entry.layer_count == 3 and entry.stage_timing is True

    def test_a_request_without_id_gets_one(self) -> None:
        live = LiveRegistry(FakeClock()).start(None, "m")

        assert live.request_id.startswith("anon-")


class TestChatEngineLiveProgress:
    def test_counts_prefill_and_tokens(self) -> None:
        live = LiveRequest("r1", "m")
        engine = ChatEngine(_AlwaysTokenOneDecoder(), eos_token_ids=set(), live=live)

        engine.generate(_request(num_predict=3))

        assert live.prefilled_tokens == 3
        assert live.generated_tokens == 3
        assert live.first_token_seconds() is not None


class TestLiveRouter:
    def test_lists_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = LiveRegistry()
        monkeypatch.setattr(LiveRegistry, "_shared", registry)
        registry.start("r1", "tiny").queued(7)
        app = FastAPI()
        app.include_router(live_router.router)

        body = TestClient(app).get("/api/live").json()

        assert body["requests"][0]["request_id"] == "r1"
        assert body["requests"][0]["phase"] == "queued"
        assert body["requests"][0]["prompt_tokens"] == 7
        assert registry.watched
