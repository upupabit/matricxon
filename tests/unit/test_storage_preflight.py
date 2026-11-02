import logging
from pathlib import Path

import pytest

from app import main
from app.cluster.storage_preflight import StoragePreflight
from app.config import Settings
from app.storage.paths import StoragePaths


class _Probe:
    def __init__(self, shared: bool) -> None:
        self._shared = shared

    def fstype(self, _path: Path) -> str:
        return "nfs4" if self._shared else "ext4"

    def is_shared(self, _path: Path) -> bool:
        return self._shared


def _preflight(tmp_path: Path, shared: bool = False, **overrides: object) -> StoragePreflight:
    values: dict[str, object] = {
        "cluster_role": "seed",
        "models_dir": tmp_path / "share/models",
        "prompt_cache_dir": tmp_path / "share/cluster/prompt_cache",
        "cache_key": "",
    } | overrides
    settings = Settings(_env_file=None, **values)
    return StoragePreflight(settings, StoragePaths(settings), _Probe(shared))


def test_a_node_outside_a_cluster_is_not_checked(tmp_path: Path) -> None:
    assert _preflight(tmp_path, cluster_role="off").problems() == []


def test_an_unmounted_share_names_both_folders_and_asks_about_the_mount(tmp_path: Path) -> None:
    problems = _preflight(tmp_path).problems()
    assert len(problems) == 2
    assert "MATRICXON_MODELS_DIR" in problems[0]
    assert "MATRICXON_PROMPT_CACHE_DIR" in problems[1]
    assert all("local 'ext4'" in p and "mounted?" in p for p in problems)


def test_a_missing_folder_on_shared_storage_has_no_mount_hint(tmp_path: Path) -> None:
    problems = _preflight(tmp_path, shared=True).problems()
    assert problems and not any("mounted?" in p for p in problems)


def test_a_mounted_share_without_the_key_reports_the_key(tmp_path: Path) -> None:
    (tmp_path / "share/models").mkdir(parents=True)
    (tmp_path / "share/cluster").mkdir()
    problems = _preflight(tmp_path, shared=True).problems()
    assert len(problems) == 1 and "prompt_cache.key" in problems[0]


def test_a_ready_node_has_no_problems(tmp_path: Path) -> None:
    (tmp_path / "share/models").mkdir(parents=True)
    (tmp_path / "share/cluster").mkdir()
    (tmp_path / "share/cluster/prompt_cache.key").write_bytes(b"k" * 32)
    assert _preflight(tmp_path, shared=True).problems() == []
    assert _preflight(tmp_path, cache_key="secret").problems() == []


def test_require_storage_exits_with_one_log_line_per_problem(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(StoragePreflight, "problems", lambda _self: ["a", "b"])
    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
        main.MatricxonApp.require_storage()
    assert [r.getMessage() for r in caplog.records] == [
        "matricxon cannot start: a",
        "matricxon cannot start: b",
    ]
