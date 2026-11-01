"""Gives every request an id (`X-Request-ID`, kept when the caller sends a sane one) and counts it
for /metrics once its last byte is sent. Plain ASGI, so streamed replies pass through untouched."""

import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.server.metrics import Metrics

HEADER = b"x-request-id"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        given = dict(scope["headers"]).get(HEADER, b"").decode("latin-1")
        request_id = given if _SAFE_ID.match(given) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.monotonic()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [h for h in message.get("headers", []) if h[0].lower() != HEADER]
                message["headers"] = [*headers, (HEADER, request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            Metrics.shared().request(scope["method"], route, status, time.monotonic() - started)
