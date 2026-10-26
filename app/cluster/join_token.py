"""One-time join token. A fresh node with no admin secret can be taken over by anyone who reaches
its port (provision it into their cluster, or set the first admin secret). So while the node is
fresh and has no secret, those two routes need this token from another machine. It is made at
first start, shown in the log and kept in the node state folder (a restart keeps it) until the node
is provisioned or gets a secret. pAIring's Add nodes form asks for it.

From this machine (loopback: pAIring managing its own Matricxon) no token is needed: anyone there
can edit .env. Nodes already in a cluster, or with a secret, are unaffected (the secret protects
them).
"""

import hmac
import logging
import os
import secrets
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

HEADER = "X-Matricxon-Join-Token"
FILE_NAME = "join_token"
_LOOPBACK = ("127.0.0.1", "::1")


class JoinToken:
    def __init__(
        self, path: Path, secret_of: Callable[[], str | None], fresh_of: Callable[[], bool]
    ) -> None:
        self._path = path
        self._secret_of = secret_of
        self._fresh_of = fresh_of

    @property
    def needed(self) -> bool:
        return not self._secret_of() and self._fresh_of()

    def announce(self) -> str | None:
        """At startup: makes the token if needed (else drops a stale one) and logs it."""
        if not self.needed:
            self.consume()
            return None
        token = self._load() or self._create()
        logger.warning(
            "This node is new and has no admin secret. To add it to a cluster from pAIring "
            "(Cluster > Add nodes), use this join token: %s",
            token,
        )
        return token

    def check(self, request: Request) -> None:
        """403 unless the request may provision or secure this node (see the module docstring)."""
        if not self.needed or (request.client is not None and request.client.host in _LOOPBACK):
            return
        expected = self._load()
        given = request.headers.get(HEADER, "")
        if not expected or not given or not hmac.compare_digest(expected, given):
            raise HTTPException(
                status_code=403,
                detail="this node is new: give its join token (shown in its log at startup)",
            )

    def consume(self) -> None:
        self._path.unlink(missing_ok=True)

    def _load(self) -> str | None:
        try:
            return self._path.read_text().strip() or None
        except FileNotFoundError:
            return None

    def _create(self) -> str:
        token = secrets.token_urlsafe(18)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(token)
        return token
