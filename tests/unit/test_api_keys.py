from starlette.requests import Request

from app.server.api_keys import ApiKeyAuth


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw})


def test_reads_bearer_and_x_api_key() -> None:
    assert ApiKeyAuth.given_key(_request({"Authorization": "Bearer abc"})) == "abc"
    assert ApiKeyAuth.given_key(_request({"Authorization": "bearer  abc "})) == "abc"
    assert ApiKeyAuth.given_key(_request({"X-API-Key": "xyz"})) == "xyz"
    assert ApiKeyAuth.given_key(_request({})) == ""


def test_enabled_only_with_a_non_empty_key() -> None:
    assert not ApiKeyAuth(lambda: "").enabled
    assert not ApiKeyAuth(lambda: " , ").enabled
    assert ApiKeyAuth(lambda: "k").enabled
