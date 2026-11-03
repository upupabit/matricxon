from pydantic import BaseModel


class VersionResponse(BaseModel):
    version: str
    commit: str | None = None  # this install's git commit, when it is a checkout
