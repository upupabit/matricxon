from fastapi import APIRouter

from app.runtime.live_requests import LiveRegistry
from app.schemas.live import LiveSnapshot

router = APIRouter()


@router.get("/api/live", response_model=LiveSnapshot)
async def get_live() -> LiveSnapshot:
    """Running and recent replies. Each call keeps stage timing on for a short while."""
    return LiveRegistry.shared().snapshot()
