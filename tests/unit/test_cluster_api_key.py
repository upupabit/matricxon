import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cluster.provisioning import EnvFile, NodeProvisioner
from app.config import settings
from app.dependencies import get_node_provisioner
from app.main import MatricxonApp
from app.server.admin_auth import NONCE_HEADER, SIGNATURE_HEADER, TIME_HEADER, AdminAuth

UNLOAD = {"model": "any:latest", "messages": [], "keep_alive": 0}


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "admin_secret", None)
    env = tmp_path / ".env"
    app = MatricxonApp().build()
    app.dependency_overrides[get_node_provisioner] = lambda: NodeProvisioner(
        None, EnvFile(env), kill=lambda: None
    )
    return TestClient(app), env


def _signed(secret: str, body: bytes) -> dict[str, str]:
    stamp, nonce = str(time.time()), f"n{time.monotonic_ns()}"
    signature = AdminAuth.sign(secret, "PUT", "/api/cluster/api-key", body, stamp, nonce)
    return {TIME_HEADER: stamp, NONCE_HEADER: nonce, SIGNATURE_HEADER: signature}


def test_setting_a_key_applies_at_once_and_is_saved(node: tuple[TestClient, Path]) -> None:
    client, env = node
    response = client.put("/api/cluster/api-key", json={"key": "k1"})
    assert response.json() == {"api_key": True}
    assert "MATRICXON_API_KEYS=k1" in env.read_text()
    assert client.post("/api/chat", json=UNLOAD).status_code == 401  # no restart needed
    ok = client.post("/api/chat", json=UNLOAD, headers={"Authorization": "Bearer k1"})
    assert ok.status_code == 200


def test_without_admin_secret_only_the_current_key_can_change_it(
    node: tuple[TestClient, Path],
) -> None:
    client, env = node
    client.put("/api/cluster/api-key", json={"key": "k1"})
    assert client.put("/api/cluster/api-key", json={"key": ""}).status_code == 401
    rotated = client.put(
        "/api/cluster/api-key", json={"key": "k2"}, headers={"Authorization": "Bearer k1"}
    )
    assert rotated.status_code == 200 and "MATRICXON_API_KEYS=k2" in env.read_text()
    removed = client.put(
        "/api/cluster/api-key", json={"key": ""}, headers={"Authorization": "Bearer k2"}
    )
    assert removed.json() == {"api_key": False}
    assert "MATRICXON_API_KEYS" not in env.read_text()
    assert client.post("/api/chat", json=UNLOAD).status_code == 200


def test_with_an_admin_secret_the_call_must_be_signed(
    node: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = node
    monkeypatch.setattr(settings, "admin_secret", "s3cret")
    body = b'{"key": "k1"}'
    headers = {"Content-Type": "application/json"}
    assert client.put("/api/cluster/api-key", content=body, headers=headers).status_code == 401
    signed = client.put(
        "/api/cluster/api-key", content=body, headers={**headers, **_signed("s3cret", body)}
    )
    assert signed.status_code == 200


def test_keys_with_spaces_or_commas_are_refused(node: tuple[TestClient, Path]) -> None:
    client, _ = node
    for bad in ("a b", "a,b"):
        assert client.put("/api/cluster/api-key", json={"key": bad}).status_code == 422


def test_a_new_node_gets_the_key_when_it_is_set_up(tmp_path: Path) -> None:
    from dataclasses import replace

    from tests.unit.test_cluster_provisioning import _node, _shared

    checker, _ = _node(tmp_path)
    provisioner = NodeProvisioner(checker, EnvFile(tmp_path / ".env"), kill=lambda: None)
    provisioner.provision(replace(_shared(tmp_path), restart=False, api_key="k9"))
    assert "MATRICXON_API_KEYS=k9" in (tmp_path / ".env").read_text()
