"""max_loaded_models counts chat models only (app.models.load_limit)."""

import json
from pathlib import Path

import pytest

from app.models import load_limit
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from tests.tiny_gguf import build_tiny_mistral3_gguf
from tests.unit.test_model_manager import FakeClock


def _install(models_dir: Path, tag: str, capabilities: list[str]) -> None:
    repo_dir = models_dir / tag.replace(":", "_")
    repo_dir.mkdir(parents=True)
    gguf_path = repo_dir / "model.gguf"
    build_tiny_mistral3_gguf(gguf_path)
    installed = InstalledModel(
        tag=tag,
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=capabilities,
        size_bytes=gguf_path.stat().st_size,
        family="mistral3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def manager(tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> ModelManager:
    models_dir = tmp_path / "models"
    for tag in ("chat-a:latest", "chat-b:latest", "chat-c:latest", "vlm:latest"):
        _install(models_dir, tag, ["completion"])
    _install(models_dir, "embed:latest", ["embedding"])
    _install(models_dir, "rerank:latest", ["completion"])
    monkeypatch.setattr(load_limit, "is_scoring_model", lambda path: "rerank" in str(path))
    return ModelManager(ModelCatalog(models_dir), max_loaded=2, clock=clock)


def _loaded(manager: ModelManager) -> set[str]:
    return {handle.tag for handle in manager.list_loaded()}


class TestLoadLimit:
    def test_helpers_load_beside_the_chat_models(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("chat-a:latest")
        clock.advance(1)
        manager.get_or_load("chat-b:latest")
        clock.advance(1)

        manager.get_or_load("embed:latest")
        manager.get_or_load("rerank:latest")

        assert _loaded(manager) == {
            "chat-a:latest",
            "chat-b:latest",
            "embed:latest",
            "rerank:latest",
        }

    def test_a_chat_model_only_pushes_out_another_chat_model(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("embed:latest")  # the least recently used of all
        clock.advance(1)
        manager.get_or_load("chat-a:latest")
        clock.advance(1)
        manager.get_or_load("chat-b:latest")
        clock.advance(1)

        manager.get_or_load("chat-c:latest")

        assert _loaded(manager) == {"embed:latest", "chat-b:latest", "chat-c:latest"}

    def test_the_vision_role_does_not_count_until_used_for_chat(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("chat-a:latest")
        clock.advance(1)
        manager.get_or_load("vlm:latest", role="vision")
        clock.advance(1)
        manager.get_or_load("chat-b:latest")
        assert _loaded(manager) == {"chat-a:latest", "vlm:latest", "chat-b:latest"}

        clock.advance(1)
        manager.get_or_load("vlm:latest")  # now a chat model: three chat models, one too many
        clock.advance(1)
        manager.get_or_load("chat-c:latest")

        assert _loaded(manager) == {"vlm:latest", "chat-c:latest"}
