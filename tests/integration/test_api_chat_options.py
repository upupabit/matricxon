import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel
from app.runtime.json_grammar import JsonGrammar

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def _chat(client: TestClient, model: InstalledModel, **extra: object) -> tuple[str, dict]:
    body = {"model": model.tag, "messages": [{"role": "user", "content": "hi"}], **extra}
    response = client.post("/api/chat", json=body)
    assert response.status_code == 200, response.text
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    return "".join(line["message"]["content"] for line in lines), lines[-1]


def test_format_json_reply_is_a_json_object(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    text, done = _chat(
        client, tiny_mistral3_model, format="json", options={"num_predict": 400, "seed": 1}
    )
    assert done["done_reason"] == "stop"
    assert isinstance(json.loads(text), dict)


def test_schema_reply_follows_the_schema(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    text, done = _chat(
        client, tiny_mistral3_model, format=SCHEMA, options={"num_predict": 120, "seed": 3}
    )
    grammar = JsonGrammar(SCHEMA)
    assert grammar.feed(text.encode())  # valid so far, even if num_predict cut it short
    if done["done_reason"] == "stop":
        assert set(json.loads(text)) == {"name", "age"}


def test_stop_text_ends_the_reply_before_it(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    greedy = {"temperature": 0, "num_predict": 40}
    full, _ = _chat(client, tiny_mistral3_model, options=greedy)
    stop = full[len(full) // 2 : len(full) // 2 + 3]
    assert stop

    text, done = _chat(client, tiny_mistral3_model, options={**greedy, "stop": [stop]})

    assert text == full[: full.index(stop)]
    assert done["done_reason"] == "stop"


def test_length_is_reported_when_num_predict_runs_out(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    _, done = _chat(client, tiny_mistral3_model, options={"temperature": 0, "num_predict": 3})
    assert done["done_reason"] in ("length", "stop")
    assert done["eval_count"] <= 3


def test_new_sampling_options_are_accepted(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    options = {"min_p": 0.1, "presence_penalty": 0.5, "frequency_penalty": 0.5, "num_predict": 5}
    _, done = _chat(client, tiny_mistral3_model, options=options)
    assert done["done"] is True


def test_out_of_range_options_are_refused(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    body = {"model": tiny_mistral3_model.tag, "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/api/chat", json={**body, "options": {"min_p": 2}})
    assert response.status_code == 422


def test_generate_takes_the_same_options_and_stop_texts(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    def run(options: dict) -> tuple[str, dict]:
        body = {"model": tiny_mistral3_model.tag, "prompt": "hi", "options": options}
        response = client.post("/api/generate", json=body)
        assert response.status_code == 200, response.text
        lines = [json.loads(line) for line in response.text.splitlines() if line]
        return "".join(line["response"] for line in lines), lines[-1]

    greedy = {"temperature": 0, "num_predict": 40, "min_p": 0.05}
    full, _ = run(greedy)
    stop = full[len(full) // 2 : len(full) // 2 + 3]
    text, done = run({**greedy, "stop": [stop]})
    assert text == full[: full.index(stop)]
    assert done["done_reason"] == "stop"
