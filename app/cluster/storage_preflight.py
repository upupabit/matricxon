from pathlib import Path

from app.cluster.fs_probe import FilesystemProbe
from app.config import Settings
from app.storage.paths import StoragePaths


class StoragePreflight:
    """Startup check of a cluster node's shared folders and cache key, so a missing mount stops the
    server with a short reason instead of a traceback. A node with cluster role "off" is not
    checked."""

    def __init__(
        self, settings: Settings, paths: StoragePaths, probe: FilesystemProbe | None = None
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._probe = probe or FilesystemProbe()

    def problems(self) -> list[str]:
        if self._settings.cluster_role == "off":
            return []
        found = [
            self._missing(path, setting)
            for path, setting in (
                (self._paths.models_dir, "MATRICXON_MODELS_DIR"),
                (self._paths.state_dir, "MATRICXON_PROMPT_CACHE_DIR"),
            )
        ]
        found.append(self._key_missing())
        return [problem for problem in found if problem]

    def _missing(self, path: Path, setting: str) -> str:
        if path.is_dir():
            return ""
        return f"{path} ({setting}) does not exist{self._mount_hint(path)}"

    def _key_missing(self) -> str:
        key_file = self._paths.cache_key_file
        if self._settings.cache_key or key_file.exists() or not key_file.parent.is_dir():
            return ""  # a missing folder is already reported
        return f"cache key {key_file} not found: copy it from the seed or set MATRICXON_CACHE_KEY"

    def _mount_hint(self, path: Path) -> str:
        if self._settings.cluster_trust_shared_paths or self._probe.is_shared(path):
            return ""
        kind = self._probe.fstype(path) or "unknown"
        return (
            f" (it resolves to a local '{kind}' filesystem): is the shared storage (NFS) mounted?"
        )
