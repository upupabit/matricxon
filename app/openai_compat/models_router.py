"""`GET /v1/models` and `POST /v1/embeddings`, OpenAI's shapes over Matricxon's catalog and
`/api/embed`'s handler."""

import asyncio
import base64
import os
import struct

from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_model_catalog, get_model_manager
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from app.openai_compat.schemas import (
    OpenAIEmbedding,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
    OpenAIModel,
    OpenAIModelList,
    Usage,
)
from app.routers.embed_router import EmbedRequestHandler
from app.schemas.embed import EmbedOptions, EmbedRequest
from app.server.api_keys import require_api_key
from app.server.errors import UnknownModelError, UnsupportedOptionError
from app.server.node_kind import require_compute

router = APIRouter()


class OpenAIModels:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    @staticmethod
    def _entry(installed: InstalledModel) -> OpenAIModel:
        try:
            created = int(os.path.getmtime(installed.path))
        except OSError:
            created = 0
        return OpenAIModel(id=installed.tag, created=created)

    def all(self) -> OpenAIModelList:
        return OpenAIModelList(data=[self._entry(m) for m in self._catalog.list_installed()])

    def one(self, model: str) -> OpenAIModel:
        found = next((m for m in self._catalog.list_installed() if m.tag == model), None)
        if found is None:
            raise UnknownModelError(f"model {model!r} is not installed")
        return self._entry(found)


class OpenAIEmbeddings:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def handle(self, request: OpenAIEmbeddingsRequest) -> OpenAIEmbeddingsResponse:
        if request.dimensions is not None:
            raise UnsupportedOptionError("dimensions: embeddings keep the model's own size")
        embed = EmbedRequest(
            model=request.model,
            input=request.input,
            options=EmbedOptions(num_ctx=settings.openai_num_ctx),
        )
        vectors = (await EmbedRequestHandler(self._model_manager).handle(embed)).embeddings
        handle = await asyncio.to_thread(self._model_manager.get_or_load, request.model)
        tokens = sum(len(handle.tokenizer.encode(text)) for text in embed.inputs())
        data = [
            OpenAIEmbedding(index=i, embedding=self._encoded(v, request.encoding_format))
            for i, v in enumerate(vectors)
        ]
        return OpenAIEmbeddingsResponse(
            data=data,
            model=request.model,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    @staticmethod
    def _encoded(vector: list[float], encoding: str) -> list[float] | str:
        if encoding == "float":
            return vector
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()


@router.get("/v1/models", dependencies=[Depends(require_api_key)], response_model=OpenAIModelList)
def get_models(catalog: ModelCatalog = Depends(get_model_catalog)) -> OpenAIModelList:
    return OpenAIModels(catalog).all()


@router.get(
    "/v1/models/{model:path}", dependencies=[Depends(require_api_key)], response_model=OpenAIModel
)
def get_model(model: str, catalog: ModelCatalog = Depends(get_model_catalog)) -> OpenAIModel:
    return OpenAIModels(catalog).one(model)


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(require_compute), Depends(require_api_key)],
    response_model=OpenAIEmbeddingsResponse,
)
async def post_embeddings(
    request: OpenAIEmbeddingsRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> OpenAIEmbeddingsResponse:
    return await OpenAIEmbeddings(model_manager).handle(request)
