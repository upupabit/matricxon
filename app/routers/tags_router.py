import logging
import threading

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_catalog
from app.models.capabilities import effective_capabilities
from app.models.catalog import ModelCatalog
from app.models.load_dtype import QUANTIZED_NATIVE_WIRED_ARCHITECTURES
from app.models.ram_estimate import estimate_ram_gb
from app.schemas.tags import ModelDetails, TagEntry, TagsResponse

router = APIRouter()
logger = logging.getLogger(__name__)


class TagsRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    def handle(self) -> TagsResponse:
        # As show_router.py: report the smaller quantized-native figure only for an architecture
        # wired to use it, never just because the setting is on globally.
        entries = [
            TagEntry(
                name=installed.tag,
                capabilities=effective_capabilities(
                    installed, self._catalog.find_paired_mmproj(installed.tag) is not None
                ),
                size=installed.size_bytes,
                details=ModelDetails(
                    family=installed.family,
                    parameter_size=installed.parameter_size,
                    context_length=installed.context_length,
                ),
                estimated_ram_gb=estimate_ram_gb(
                    installed.path,
                    settings.enable_quantized_native_compute
                    and installed.architecture in QUANTIZED_NATIVE_WIRED_ARCHITECTURES,
                    installed.architecture,
                ),
            )
            for installed in self._catalog.list_installed()
        ]
        return TagsResponse(models=entries)

    def warm_up(self) -> None:
        """Fills the per-file caches in the background at startup, so the first /api/tags calls
        after a restart don't each read every model header (slow on NFS with a cold cache)."""
        threading.Thread(target=self._warm, name="tags-warm-up", daemon=True).start()

    def _warm(self) -> None:
        try:
            self.handle()
        except Exception:
            logger.warning("model list warm-up failed", exc_info=True)


@router.get("/api/tags", response_model=TagsResponse)
def get_tags(catalog: ModelCatalog = Depends(get_model_catalog)) -> TagsResponse:
    return TagsRequestHandler(catalog).handle()
