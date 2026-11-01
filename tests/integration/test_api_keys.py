import pytest
from fastapi.testclient import TestClient

from app.config import settings

UNLOAD = {"model": "any:latest", "messages": [], "keep_alive": 0}  # answers at once, loads nothing


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", "first-key, second-key")


def test_without_keys_configured_everything_is_open(client: TestClient) -> None:
    assert client.post("/api/chat", json=UNLOAD).status_code == 200


def test_model_routes_need_a_key(client: TestClient, keys: None) -> None:
    response = client.post("/api/chat", json=UNLOAD)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    for path in ("/api/generate", "/api/embed", "/api/embeddings", "/api/rerank"):
        assert client.post(path, json={}).status_code == 401, path


def test_a_wrong_key_is_refused(client: TestClient, keys: None) -> None:
    headers = {"Authorization": "Bearer nope"}
    assert client.post("/api/chat", json=UNLOAD, headers=headers).status_code == 401


def test_any_configured_key_works_either_way(client: TestClient, keys: None) -> None:
    bearer = {"Authorization": "Bearer second-key"}
    assert client.post("/api/chat", json=UNLOAD, headers=bearer).status_code == 200
    header = {"X-API-Key": "first-key"}
    assert client.post("/api/chat", json=UNLOAD, headers=header).status_code == 200


def test_listing_and_health_stay_open(client: TestClient, keys: None) -> None:
    assert client.get("/api/tags").status_code == 200
    assert client.get("/api/health").status_code == 200
