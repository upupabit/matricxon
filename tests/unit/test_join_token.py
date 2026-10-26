"""The one-time join token (app/cluster/join_token.py): while a node is fresh and has no admin
secret, provisioning it or setting its first secret from another machine needs the token from its
log."""

import logging
import stat
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.cluster.freshness import FreshnessChecker
from app.cluster.join_token import HEADER, JoinToken
from app.cluster.provisioning import EnvFile, NodeProvisioner
from app.config import Settings
from app.dependencies import get_freshness_checker, get_join_token, get_node_provisioner
from app.main import MatricxonApp
from app.storage.paths import StoragePaths


def _request(client_ip: str, token: str | None = None) -> Request:
    headers = [(HEADER.lower().encode(), token.encode())] if token else []
    return Request({"type": "http", "headers": headers, "client": (client_ip, 5000)})


def _token(tmp_path: Path, secret: str | None = None, fresh: bool = True) -> JoinToken:
    return JoinToken(tmp_path / "node" / "join_token", lambda: secret, lambda: fresh)


def test_a_new_node_logs_a_private_token_that_survives_a_restart(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    token = _token(tmp_path).announce()

    assert token and token in caplog.text
    path = tmp_path / "node" / "join_token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _token(tmp_path).announce() == token


def test_another_machine_needs_the_token_and_this_machine_does_not(tmp_path) -> None:
    joiner = _token(tmp_path)
    token = joiner.announce()

    joiner.check(_request("10.0.0.7", token))
    joiner.check(_request("127.0.0.1"))
    for bad in (None, "wrong-token"):
        with pytest.raises(HTTPException) as refused:
            joiner.check(_request("10.0.0.7", bad))
        assert refused.value.status_code == 403


def test_a_node_with_a_secret_or_a_history_needs_no_token(tmp_path) -> None:
    (tmp_path / "node").mkdir()
    (tmp_path / "node" / "join_token").write_text("stale")
    for node in (_token(tmp_path, secret="s"), _token(tmp_path, fresh=False)):
        node.check(_request("10.0.0.7"))
        assert node.announce() is None
    assert not (tmp_path / "node" / "join_token").exists()


def test_the_token_file_does_not_make_a_node_look_used(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        prompt_cache_dir=tmp_path / "state" / "prompt_cache",
        node_state_dir=tmp_path / "node",
    )
    checker = FreshnessChecker(settings, StoragePaths(settings))
    JoinToken(tmp_path / "node" / "join_token", lambda: None, lambda: True).announce()

    assert checker.report().fresh


def test_provisioning_from_another_machine_needs_the_token_and_uses_it_up(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        prompt_cache_dir=tmp_path / "state" / "prompt_cache",
        node_state_dir=tmp_path / "node",
    )
    checker = FreshnessChecker(settings, StoragePaths(settings))
    joiner = JoinToken(tmp_path / "node" / "join_token", lambda: None, lambda: True)
    token = joiner.announce()
    app = MatricxonApp().build()
    app.dependency_overrides[get_freshness_checker] = lambda: checker
    app.dependency_overrides[get_join_token] = lambda: joiner
    app.dependency_overrides[get_node_provisioner] = lambda: NodeProvisioner(
        checker, EnvFile(tmp_path / ".env"), kill=lambda: None
    )
    (tmp_path / "shared" / "models").mkdir(parents=True)
    (tmp_path / "shared" / "cache").mkdir(parents=True)
    (tmp_path / "shared" / "prompt_cache.key").write_bytes(b"k" * 32)
    body = {
        "models_dir": str(tmp_path / "shared" / "models"),
        "prompt_cache_dir": str(tmp_path / "shared" / "cache"),
        "restart": False,
    }
    client = TestClient(app)

    assert client.post("/api/cluster/provision", json=body).status_code == 403
    assert client.put("/api/cluster/admin-secret", json={"secret": "s"}).status_code == 403
    ok = client.post("/api/cluster/provision", json=body, headers={HEADER: token})
    assert ok.status_code == 200
    assert not (tmp_path / "node" / "join_token").exists()
