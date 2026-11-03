"""Updating a node from its own upstream (app/update): fetch, fast-forward, pip when needed, then a
restart. Real git repositories in tmp_path; pip and the restart are fakes."""

import subprocess
import threading
from pathlib import Path

import pytest

from app.update.git_checkout import GitCheckout, Source
from app.update.self_update import SelfUpdater, UpdateRefusedError


def _git(folder: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _commit(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text)
    _git(folder, "add", name)
    _git(folder, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name)


@pytest.fixture
def repos(tmp_path):
    """(the node's checkout, a second clone to push new commits from)."""
    upstream = tmp_path / "upstream.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(upstream))
    author = tmp_path / "author"
    _git(tmp_path, "clone", "-q", str(upstream), str(author))
    _git(author, "checkout", "-qb", "main")
    _commit(author, "requirements.txt", "a\n")
    _git(author, "push", "-q", "-u", "origin", "main")
    node = tmp_path / "node"
    _git(tmp_path, "clone", "-q", str(upstream), str(node))
    return node, author


class _Calls:
    def __init__(self) -> None:
        self.pip: list[Path] = []
        self.restarted = threading.Event()


def _updater(node: Path, calls: _Calls, **kwargs) -> SelfUpdater:
    updater = SelfUpdater(
        calls.restarted.set,
        git=GitCheckout(node),
        pip=calls.pip.append,
        in_container=lambda: False,
        **kwargs,
    )
    updater.RESTART_DELAY_SECONDS = 0
    return updater


def test_new_commits_are_applied_then_the_node_restarts(repos):
    node, author = repos
    _commit(author, "requirements.txt", "a\nb\n")
    _git(author, "push", "-q")
    calls = _Calls()
    updater = _updater(node, calls)
    updater._status.state = "updating"

    updater.run()

    status = updater.status()
    assert status["state"] == "restarting" and status["error"] is None
    assert status["commit_after"] == _git(author, "rev-parse", "--short", "HEAD")
    assert (node / "requirements.txt").read_text() == "a\nb\n"
    assert len(calls.pip) == 1 and calls.restarted.wait(2)


def test_no_new_commits_means_up_to_date_and_no_restart(repos):
    node, _author = repos
    calls = _Calls()
    updater = _updater(node, calls)

    updater.run()

    assert updater.status()["state"] == "up_to_date"
    assert not calls.restarted.wait(0.2) and not calls.pip


def test_unchanged_requirements_skip_pip(repos):
    node, author = repos
    _commit(author, "app.py", "x = 1\n")
    _git(author, "push", "-q")
    calls = _Calls()
    updater = _updater(node, calls)

    updater.run()

    assert updater.status()["state"] == "restarting" and not calls.pip


def test_a_local_change_in_the_way_is_an_error_and_nothing_restarts(repos):
    node, author = repos
    _commit(author, "requirements.txt", "new\n")
    _git(author, "push", "-q")
    (node / "requirements.txt").write_text("edited here\n")
    calls = _Calls()
    updater = _updater(node, calls)

    updater.run()

    status = updater.status()
    assert status["state"] == "error" and "merge" in status["error"]
    assert not calls.restarted.wait(0.2)


def test_a_container_or_a_copy_without_git_is_refused(tmp_path):
    calls = _Calls()
    in_container = SelfUpdater(
        calls.restarted.set, git=GitCheckout(tmp_path), in_container=lambda: True
    )
    with pytest.raises(UpdateRefusedError, match="container"):
        in_container.start()

    no_git = SelfUpdater(calls.restarted.set, git=GitCheckout(tmp_path), in_container=lambda: False)
    with pytest.raises(UpdateRefusedError, match="git checkout"):
        no_git.start()
    assert no_git.status()["state"] == "idle"


def test_a_second_start_while_updating_is_refused(repos):
    node, _author = repos
    updater = _updater(node, _Calls())
    updater._status.state = "updating"

    with pytest.raises(UpdateRefusedError, match="already running"):
        updater.start()


class TestUpdateRoutes:
    @pytest.fixture
    def client(self, monkeypatch, repos):
        from fastapi.testclient import TestClient

        from app.config import settings
        from app.dependencies import get_self_updater
        from app.main import MatricxonApp

        monkeypatch.setattr(settings, "admin_secret", None)  # this machine's .env may set one
        self.updater = _updater(repos[0], _Calls())
        app = MatricxonApp().build()
        app.dependency_overrides[get_self_updater] = lambda: self.updater
        return TestClient(app)

    def test_start_then_status(self, client):
        assert client.post("/api/update").status_code == 202
        self.updater._status.state = "up_to_date"
        assert client.get("/api/update").json()["state"] == "up_to_date"

    def test_a_refused_start_is_409(self, client):
        self.updater._status.state = "updating"
        response = client.post("/api/update")
        assert response.status_code == 409 and "already running" in response.json()["detail"]

    def test_an_admin_secret_is_checked(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "admin_secret", "s3cret")
        assert client.post("/api/update").status_code in (401, 403)

    def test_the_version_names_this_checkout_commit(self, client):
        assert "commit" in client.get("/api/version").json()


class TestNamedSource:
    @pytest.fixture
    def partner(self, repos, tmp_path):
        """A partner repository: the same history plus a private commit."""
        _node, author = repos
        partner = tmp_path / "partner.git"
        _git(tmp_path, "clone", "-q", "--bare", str(author), str(partner))
        work = tmp_path / "partner-work"
        _git(tmp_path, "clone", "-q", str(partner), str(work))
        _commit(work, "private.py", "secret_feature = True\n")
        _git(work, "push", "-q", "origin", "HEAD:main")
        return partner

    def test_the_named_repository_is_fetched_and_origin_stays(self, repos, partner):
        node, _author = repos
        origin = _git(node, "remote", "get-url", "origin")
        calls = _Calls()
        updater = _updater(node, calls, has_admin_secret=lambda: True)
        updater._git.use(Source(f"file://{partner}", "main"))

        updater.run()

        assert updater.status()["state"] == "restarting"
        assert (node / "private.py").exists()
        assert _git(node, "remote", "get-url", "origin") == origin
        assert _git(node, "remote", "get-url", "update-source") == f"file://{partner}"

    def test_a_named_repository_needs_the_admin_secret(self, repos):
        node, _author = repos
        updater = _updater(node, _Calls())

        with pytest.raises(UpdateRefusedError, match="MATRICXON_ADMIN_SECRET"):
            updater.start(Source("https://example.com/partner/matricxon.git", "main"))

    def test_the_token_goes_only_into_git_s_environment(self):
        env = Source("https://github.com/p/m.git", "main", token="t0k").env()

        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/p/m.git.extraHeader"
        assert env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
        assert "t0k" not in env["GIT_CONFIG_KEY_0"]

    def test_a_url_with_credentials_is_refused(self):
        from pydantic import ValidationError

        from app.schemas.update import UpdateSourceRequest

        with pytest.raises(ValidationError):
            UpdateSourceRequest(url="https://user:pw@github.com/p/m.git")
        with pytest.raises(ValidationError):
            UpdateSourceRequest(url="https://github.com/p/m.git", branch="--upload-pack=x")


def test_a_refused_private_repository_says_what_to_check():
    from app.update.git_checkout import GitError

    assert "access token" in str(GitError("git fetch failed: fatal: could not read Username"))
