from pydantic import BaseModel, Field


class CachePersistenceStatus(BaseModel):
    enabled: bool
    budget_mb: int
    ttl_hours: int
    save_every_seconds: int = 30
    files: int
    # Distinct chats with a saved copy (`files` can be more: branches of one chat).
    chats: int = 0
    used_bytes: int
    folder: str | None = None
    shared: bool = False


class CachePersistenceUpdate(BaseModel):
    """Omitted fields stay unchanged. Turning `enabled` off also deletes every stored cache."""

    enabled: bool | None = None
    budget_mb: int | None = Field(default=None, ge=0)
    ttl_hours: int | None = Field(default=None, ge=1)
    # How often a chat still in RAM is copied to disk after a reply; 0: only when it leaves RAM.
    save_every_seconds: int | None = Field(default=None, ge=0, le=3600)
