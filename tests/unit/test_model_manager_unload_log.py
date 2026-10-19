"""Each unload is logged with its reason: keep-alive ran out, unload request, or making room."""

import logging
from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from tests.unit.test_model_manager import FakeClock, _install_tiny_model


@pytest.fixture
def setup(tmp_path: Path) -> tuple[ModelManager, FakeClock]:
    models_dir = tmp_path / "models"
    for tag in ("model-a:latest", "model-b:latest", "model-c:latest"):
        _install_tiny_model(models_dir, tag)
    clock = FakeClock()
    manager = ModelManager(
        ModelCatalog(models_dir), max_loaded=2, default_keep_alive_seconds=300, clock=clock
    )
    return manager, clock


def _unload_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("unloading ")]


def test_an_expired_keep_alive_is_logged_with_the_times(setup, caplog) -> None:
    manager, clock = setup
    manager.get_or_load("model-a:latest")
    clock.advance(301)
    with caplog.at_level(logging.INFO, logger="app.models.loaded_models"):
        manager.list_loaded()
    assert _unload_lines(caplog) == ["unloading model-a:latest: idle 5 min, keep-alive 5 min"]


def test_an_unload_request_and_making_room_are_logged(setup, caplog) -> None:
    manager, clock = setup
    manager.get_or_load("model-a:latest")
    clock.advance(1)
    manager.get_or_load("model-b:latest")
    with caplog.at_level(logging.INFO, logger="app.models.loaded_models"):
        manager.get_or_load("model-c:latest")
        manager.unload("model-b:latest")
    assert _unload_lines(caplog) == [
        "unloading model-a:latest: making room (at most 2 loaded)",
        "unloading model-b:latest: unload request",
    ]
