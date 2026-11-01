from typing import Literal

from pydantic import BaseModel, Field


class ReadinessCheckInfo(BaseModel):
    id: str
    status: str
    what: str
    why: str
    setting: str | None
    size_bytes: int
    hint: str


class ReadinessResponse(BaseModel):
    role: str
    node_id: str
    ready: bool
    active: bool
    checks: list[ReadinessCheckInfo]
    nodes: list[str]
    node_kind: str = "compute"
    admin_auth: bool = False


class FreshnessResponse(BaseModel):
    fresh: bool
    reasons: list[str]


class ProvisionBody(BaseModel):
    models_dir: str
    prompt_cache_dir: str
    restart: bool = True
    advertise_url: str | None = None
    consensus_seeds: dict[str, str] | None = None
    admin_secret: str | None = None
    api_key: str | None = Field(default=None, max_length=200, pattern=r"^[^\s,]*$")
    node_kind: Literal["compute", "voter"] | None = None


class ProvisionResponse(BaseModel):
    restarting: bool


class ConfigureBody(BaseModel):
    role: Literal["off", "seed"]
    models_dir: str | None = None
    prompt_cache_dir: str | None = None
    advertise_url: str | None = None
    restart: bool = True
    admin_secret: str | None = None
    node_kind: Literal["compute", "voter"] | None = None


class AdminSecretBody(BaseModel):
    secret: str = ""  # "" removes it (the routes become open)


class AdminSecretResponse(BaseModel):
    admin_auth: bool


class ApiKeyBody(BaseModel):
    # The key for the routes that run a model (MATRICXON_API_KEYS); "" removes it.
    key: str = Field(default="", max_length=200, pattern=r"^[^\s,]*$")


class ApiKeyResponse(BaseModel):
    api_key: bool
