from fastapi import APIRouter

from app.config import MATRICXON_VERSION
from app.schemas.version import VersionResponse
from app.update.git_checkout import GitCheckout, GitError
from app.update.self_update import PROJECT_DIR

router = APIRouter()


def _commit() -> str | None:
    """Read once: a self-update restarts the process, so it is always this code's commit."""
    try:
        return GitCheckout(PROJECT_DIR).head() if GitCheckout(PROJECT_DIR).is_checkout() else None
    except (GitError, OSError):
        return None


_COMMIT = _commit()


@router.get("/api/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=MATRICXON_VERSION, commit=_COMMIT)
