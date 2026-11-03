import re

from pydantic import BaseModel, field_validator


class UpdateSourceRequest(BaseModel):
    """A repository to update from instead of the checkout's upstream (admin secret needed)."""

    url: str
    branch: str = "main"
    token: str | None = None

    @field_validator("url")
    @classmethod
    def _plain_url(cls, url: str) -> str:
        if url.startswith("https://") and "@" not in url.split("/")[2]:
            return url
        if re.match(r"^(?:ssh://)?[\w.-]+@[\w.-]+[:/][\w./~-]+$", url):
            return url
        raise ValueError("an https:// URL without credentials, or an SSH address")

    @field_validator("branch")
    @classmethod
    def _plain_branch(cls, branch: str) -> str:
        if not re.match(r"^[\w./-]+$", branch) or branch.startswith("-"):
            raise ValueError("a branch or tag name")
        return branch


class UpdateStatusResponse(BaseModel):
    """A self-update of this node (app/update/self_update.py)."""

    state: str
    step: int
    total_steps: int
    label: str
    percent: int | None = None
    error: str | None = None
    commit_before: str | None = None
    commit_after: str | None = None
