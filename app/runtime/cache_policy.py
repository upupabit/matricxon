import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from app.runtime.policy_store import FilePolicyStore, PolicyStore

logger = logging.getLogger(__name__)


@dataclass
class PersistencePolicy:
    """Admin-editable limits of the on-disk prompt cache. Changes are saved to `path` (via a
    `PolicyStore`) and survive a restart; the .env `Settings` values are only first-run defaults."""

    enabled: bool
    budget_mb: int
    ttl_hours: int
    store: PolicyStore | None = None
    # A chat in RAM is copied to disk after a reply at most this often; 0: only when it leaves RAM.
    save_every_seconds: int = 30

    @property
    def budget_bytes(self) -> int:
        return self.budget_mb << 20

    @property
    def ttl_seconds(self) -> float:
        return self.ttl_hours * 3600.0

    @classmethod
    def load(
        cls, path: Path, enabled: bool, budget_mb: int, ttl_hours: int, save_every_seconds: int = 30
    ) -> "PersistencePolicy":
        """The saved policy if `path` holds a valid one, else the given defaults."""
        return cls.from_store(
            FilePolicyStore(path), enabled, budget_mb, ttl_hours, save_every_seconds
        )

    @classmethod
    def from_store(
        cls,
        store: PolicyStore,
        enabled: bool,
        budget_mb: int,
        ttl_hours: int,
        save_every_seconds: int = 30,
    ) -> "PersistencePolicy":
        policy = cls(enabled, budget_mb, ttl_hours, store, save_every_seconds)
        try:
            saved = store.read()
            if saved is not None:
                policy.enabled = bool(saved["enabled"])
                policy.budget_mb = max(0, int(saved["budget_mb"]))
                policy.ttl_hours = max(1, int(saved["ttl_hours"]))
                policy.save_every_seconds = max(
                    0, int(saved.get("save_every_seconds", save_every_seconds))
                )
        except (ValueError, KeyError, TypeError, OSError):
            logger.warning("ignoring unreadable cache policy in %s", type(store).__name__)
        return policy

    def update(
        self,
        enabled: bool | None = None,
        budget_mb: int | None = None,
        ttl_hours: int | None = None,
        save_every_seconds: int | None = None,
    ) -> None:
        if save_every_seconds is not None:
            self.save_every_seconds = max(0, save_every_seconds)
        if enabled is not None:
            self.enabled = enabled
        if budget_mb is not None:
            self.budget_mb = max(0, budget_mb)
        if ttl_hours is not None:
            self.ttl_hours = max(1, ttl_hours)
        self.save()

    def save(self) -> None:
        if self.store is None:
            return
        self.store.write({k: v for k, v in asdict(self).items() if k != "store"})
