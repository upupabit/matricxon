import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.server.error_content import ErrorContent

logger = logging.getLogger(__name__)


class MatricxonError(Exception):
    """Base class for all matricxon domain errors."""

    status_code: int = 500
    # Stable and machine-readable, sent next to the message (the message may change wording).
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownModelError(MatricxonError):
    status_code = 404
    code = "model_not_found"


class ModelInUseError(MatricxonError):
    """A model can't be removed because something still depends on it (its aliases)."""

    status_code = 409
    code = "model_in_use"


class NotLeaseHolderError(MatricxonError):
    """A cluster job (pulling models, cleaning the cache) that only the lease owner may do."""

    status_code = 409
    code = "not_lease_holder"


class UnsupportedArchitectureError(MatricxonError):
    status_code = 400
    code = "unsupported_architecture"


class UnsupportedQuantTypeError(MatricxonError):
    status_code = 400
    code = "unsupported_quant_type"


class WrongModelKindError(MatricxonError):
    """A request for the wrong kind of model, e.g. a chat with a reranker."""

    status_code = 400
    code = "wrong_model_kind"


class InvalidGGUFError(MatricxonError):
    status_code = 400
    code = "invalid_gguf"


class MissingMetadataError(MatricxonError):
    status_code = 400
    code = "missing_metadata"


class PromptTooLongError(MatricxonError):
    status_code = 400
    code = "prompt_too_long"


class ModelResolutionError(MatricxonError):
    status_code = 404
    code = "model_resolution_failed"


class PullConnectionError(MatricxonError):
    status_code = 502
    code = "pull_connection_failed"


class PullCancelledError(MatricxonError):
    """Raised when a pull is aborted because its client disconnected mid-download
    (app.pull.download_coordinator says why an abandoned download running on invisibly is a real
    problem). Reaches PullJob.run's `except MatricxonError` like every pull failure, though nobody
    listens on an already-disconnected stream by then."""

    status_code = 499
    code = "pull_cancelled"


class UnsupportedOptionError(MatricxonError):
    """An option this model or endpoint can't honour (e.g. format on an always-thinking model)."""

    status_code = 400
    code = "unsupported_option"


class UnsupportedChatRoleError(MatricxonError):
    status_code = 400
    code = "unsupported_chat_role"


class PushNotSupportedError(MatricxonError):
    status_code = 501
    code = "push_not_supported"


class InsufficientMemoryError(MatricxonError):
    status_code = 503
    code = "out_of_memory"


class DeviceUnavailableError(MatricxonError):
    """`MATRICXON_DEVICE` names a device this machine doesn't have, or the model can't run on it."""

    status_code = 503
    code = "device_unavailable"


class ErrorHandlerRegistrar:
    """Every error answers in one shape: `{"error": <message>, "code": <stable code>}` (pAIring
    reads `error` as text, so it stays a string). The OpenAI-compatible /v1 routes answer in
    OpenAI's shape instead (app.server.error_content)."""

    def register(self, app: FastAPI) -> None:
        @app.exception_handler(MatricxonError)
        async def handle_matricxon_error(request: Request, exc: MatricxonError) -> JSONResponse:
            return ErrorContent.response(request, exc.status_code, exc.message, exc.code)

        @app.exception_handler(StarletteHTTPException)
        async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
            message = exc.detail if isinstance(exc.detail, str) else "request refused"
            code = ErrorContent.HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
            # `detail` stays for callers that read FastAPI's usual field.
            return ErrorContent.response(
                request, exc.status_code, message, code, {"detail": exc.detail}, exc.headers
            )

        @app.exception_handler(RequestValidationError)
        async def handle_invalid_request(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            errors = jsonable_encoder(exc.errors())
            return ErrorContent.response(
                request, 422, ErrorContent.summary(errors), "invalid_request", {"detail": errors}
            )

        # Catches anything that isn't one of our typed errors (a bug: bad GGUF metadata shape, mmap
        # lifecycle issue, ...), which would reach Starlette's ServerErrorMiddleware and return a
        # bare non-JSON "Internal Server Error". pAIring's matricxon_client.py expects
        # `{"error":...}` and showed "Matricxon returned HTTP 500 with no error detail" while the
        # traceback sat in this server's log. Logs the traceback before responding, since this
        # handler intercepts the exception before Starlette's default logging.
        @app.exception_handler(Exception)
        async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
            logger.exception("Unhandled exception in request")
            return ErrorContent.response(request, 500, f"Internal error: {exc}", "internal_error")
