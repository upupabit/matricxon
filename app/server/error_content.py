"""What an error answer contains: Matricxon's `{"error", "code"}`, or OpenAI's
`{"error": {"message", "type", "param", "code"}}` on the /v1 routes."""

from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

OPENAI_PREFIX = "/v1/"


class ErrorContent:
    HTTP_CODES = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "too_large",
        429: "too_many_requests",
        503: "unavailable",
    }

    @staticmethod
    def openai_type(status: int) -> str:
        if status == 401:
            return "authentication_error"
        if status == 403:
            return "permission_error"
        if status == 429:
            return "rate_limit_error"
        return "api_error" if status >= 500 else "invalid_request_error"

    @classmethod
    def openai(cls, status: int, message: str, code: str) -> dict[str, Any]:
        return {
            "error": {
                "message": message,
                "type": cls.openai_type(status),
                "param": None,
                "code": code,
            }
        }

    @classmethod
    def describe(cls, error: Exception) -> tuple[int, str, str]:
        """(status, message, code) for any exception, typed or not."""
        status = int(getattr(error, "status_code", 500))
        message = getattr(error, "message", None) or getattr(error, "detail", None) or str(error)
        code = getattr(error, "code", None) or cls.HTTP_CODES.get(status, "internal_error")
        return status, str(message), str(code)

    @staticmethod
    def summary(errors: list[dict[str, Any]]) -> str:
        """One readable line for a request that failed validation."""
        parts = [
            f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'body'}: {e.get('msg', 'invalid')}"
            for e in errors[:3]
        ]
        return "invalid request: " + "; ".join(parts)

    @classmethod
    def response(
        cls,
        request: Request,
        status: int,
        message: str,
        code: str,
        extra: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> JSONResponse:
        if request.url.path.startswith(OPENAI_PREFIX):
            body = cls.openai(status, message, code)
        else:
            body = {"error": message, "code": code, **(extra or {})}
        return JSONResponse(status_code=status, content=body, headers=dict(headers or {}))
