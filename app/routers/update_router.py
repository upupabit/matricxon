from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_self_updater
from app.schemas.update import UpdateSourceRequest, UpdateStatusResponse
from app.server.admin_auth import require_admin
from app.update.git_checkout import Source
from app.update.self_update import SelfUpdater, UpdateRefusedError

router = APIRouter()


@router.post(
    "/api/update",
    dependencies=[Depends(require_admin)],
    response_model=UpdateStatusResponse,
    status_code=202,
)
def start_update(
    body: UpdateSourceRequest | None = None, updater: SelfUpdater = Depends(get_self_updater)
) -> UpdateStatusResponse:
    """Updates THIS node from its own upstream, or from `body`'s repository, then restarts it. 409
    when it can't."""
    source = Source(body.url, body.branch, body.token) if body else None
    try:
        return UpdateStatusResponse(**updater.start(source))
    except UpdateRefusedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/update", dependencies=[Depends(require_admin)], response_model=UpdateStatusResponse
)
def update_status(updater: SelfUpdater = Depends(get_self_updater)) -> UpdateStatusResponse:
    return UpdateStatusResponse(**updater.status())
