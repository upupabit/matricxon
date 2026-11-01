import base64
import json
import struct

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _body(model: InstalledModel, **extra: object) -> dict:
    return {"model": model.tag, "messages": [{"role": "user", "content": "hi"}], **extra}


def _events(text: str) -> list[dict | str]:
    out: list[dict | str] = []
    for block in text.strip().split("\n\n"):
        data = block.removeprefix("data: ")
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


class TestChatCompletions:
    def test_whole_reply(self, client: TestClient, tiny_mistral3_model: InstalledModel) -> None:
        response = client.post(
            "/v1/chat/completions", json=_body(tiny_mistral3_model, max_tokens=5, temperature=0)
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["object"] == "chat.completion" and data["id"].startswith("chatcmpl-")
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length")
        usage = data["usage"]
        assert usage["completion_tokens"] <= 5
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_streamed_reply(self, client: TestClient, tiny_mistral3_model: InstalledModel) -> None:
        body = _body(
            tiny_mistral3_model, max_tokens=5, stream=True, stream_options={"include_usage": True}
        )
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        assert events[-1] == "[DONE]"
        chunks = [e for e in events if isinstance(e, dict)]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
        assert finishes[-1] in ("stop", "length")
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] <= 5

    def test_json_object_response_format(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        body = _body(
            tiny_mistral3_model, response_format={"type": "json_object"}, max_tokens=400, seed=1
        )
        data = client.post("/v1/chat/completions", json=body).json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert isinstance(json.loads(data["choices"][0]["message"]["content"]), dict)

    def test_stop_and_extra_options(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        body = _body(tiny_mistral3_model, max_tokens=5, stop="zzzz", top_k=10, min_p=0.05)
        assert client.post("/v1/chat/completions", json=body).status_code == 200


class TestErrors:
    def test_unknown_model_in_openai_shape(self, client: TestClient) -> None:
        body = {"model": "nope:latest", "messages": [{"role": "user", "content": "hi"}]}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "model_not_found" and error["type"] == "invalid_request_error"
        assert error["message"]

    def test_unsupported_option(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post("/v1/chat/completions", json=_body(tiny_mistral3_model, n=2))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_option"

    def test_invalid_body(self, client: TestClient) -> None:
        response = client.post("/v1/chat/completions", json={"model": "x"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_image_links_are_refused(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        part = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
        body = {"model": tiny_mistral3_model.tag, "messages": [{"role": "user", "content": [part]}]}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400


class TestModelsAndEmbeddings:
    def test_models(self, client: TestClient, tiny_mistral3_model: InstalledModel) -> None:
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        assert tiny_mistral3_model.tag in [m["id"] for m in data["data"]]
        one = client.get(f"/v1/models/{tiny_mistral3_model.tag}")
        assert one.status_code == 200 and one.json()["id"] == tiny_mistral3_model.tag
        assert client.get("/v1/models/hf.co/none/x:latest").status_code == 404

    def test_embeddings(self, client: TestClient, tiny_bert_model: InstalledModel) -> None:
        body = {"model": tiny_bert_model.tag, "input": ["one", "two"]}
        data = client.post("/v1/embeddings", json=body).json()
        assert [d["index"] for d in data["data"]] == [0, 1]
        vector = data["data"][0]["embedding"]
        assert data["usage"]["prompt_tokens"] > 0

        packed = client.post("/v1/embeddings", json={**body, "encoding_format": "base64"}).json()[
            "data"
        ][0]["embedding"]
        raw = base64.b64decode(packed)
        decoded = struct.unpack(f"<{len(raw) // 4}f", raw)
        assert len(decoded) == len(vector)
        assert abs(decoded[0] - vector[0]) < 1e-5
