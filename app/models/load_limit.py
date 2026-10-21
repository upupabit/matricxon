"""Which models count toward `max_loaded_models`: chat models only.

Embedding and reranking models, and a model asked for in the vision role (pAIring's default vision
model, used for an image attachment), are helpers: they load beside the chat models instead of
pushing one out. The memory check before every load (`ensure_enough_memory_to_load`) still guards
RAM, and keep-alive still unloads idle helpers.
"""

from app.architectures.registry import is_scoring_model
from app.models.catalog import ModelCatalog


class ModelRole:
    """The `model_role` a chat request may carry (ChatRequest.model_role)."""

    VISION = "vision"


class LoadLimit:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog
        self._helper_kind: dict[str, bool] = {}  # tag -> embedding/reranker, read once

    def counts(self, tag: str, role: str | None = None) -> bool:
        """True when loading `tag` for this request counts toward the limit."""
        if role == ModelRole.VISION:
            return False
        return not self._is_helper_kind(tag)

    def _is_helper_kind(self, tag: str) -> bool:
        if tag not in self._helper_kind:
            try:
                installed = self._catalog.get(tag)
                helper = "embedding" in installed.capabilities or is_scoring_model(
                    self._catalog.gguf_path_for_tag(tag)
                )
            except Exception:  # noqa: BLE001 - unknown tag: count it; the load reports the real error
                return True
            self._helper_kind[tag] = helper
        return self._helper_kind[tag]
