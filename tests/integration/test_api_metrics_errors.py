import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.models.installed_model import InstalledModel

UNLOAD = {"model": "any:latest", "messages": [], "keep_alive": 0}


class TestRequestId:
    def test_every_answer_has_one(self, client: TestClient) -> None:
        first = client.get("/api/version").headers["X-Request-ID"]
        second = client.get("/api/version").headers["X-Request-ID"]
        assert first and second and first != second

    def test_a_sane_given_id_is_kept_and_a_strange_one_replaced(self, client: TestClient) -> None:
        kept = client.get("/api/version", headers={"X-Request-ID": "trace-42"})
        assert kept.headers["X-Request-ID"] == "trace-42"
        replaced = client.get("/api/version", headers={"X-Request-ID": "bad id\twith spaces"})
        assert replaced.headers["X-Request-ID"] != "bad id\twith spaces"


class TestMetrics:
    def test_counts_requests_by_route_and_tokens(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        body = {
            "model": tiny_mistral3_model.tag,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 3},
        }
        assert client.post("/api/chat", json=body).status_code == 200
        text = client.get("/metrics").text

        assert 'matricxon_requests_total{method="POST",route="/api/chat",status="200"}' in text
        generated = next(
            line
            for line in text.splitlines()
            if line.startswith("matricxon_generated_tokens_total ")
        )
        assert int(generated.split()[1]) >= 1
        assert "# TYPE matricxon_active_replies gauge" in text
        assert "matricxon_loaded_models 1" in text

    def test_needs_a_key_when_keys_are_set(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "api_keys", "k")
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer k"}).status_code == 200


class TestErrorShape:
    def test_typed_error_has_message_and_code(self, client: TestClient) -> None:
        body = {"model": "nope:latest", "messages": [{"role": "user", "content": "hi"}]}
        data = client.post("/api/chat", json=body).json()
        assert data["code"] == "model_not_found" and isinstance(data["error"], str)

    def test_invalid_body_keeps_detail_too(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"messages": []})
        data = response.json()
        assert response.status_code == 422
        assert data["code"] == "invalid_request" and data["error"].startswith("invalid request")
        assert isinstance(data["detail"], list)

    def test_http_errors_use_the_same_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        method = client.get("/api/chat").json()
        assert method["code"] == "method_not_allowed" and method["error"]
        monkeypatch.setattr(settings, "api_keys", "k")
        refused = client.post("/api/chat", json=UNLOAD)
        assert refused.json()["code"] == "unauthorized"
        assert refused.json()["detail"] == refused.json()["error"]
