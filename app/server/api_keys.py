import hashlib
import hmac
from collections.abc import Callable

from fastapi import HTTPException, Request

from app.config import settings


class ApiKeyAuth:
    """Optional keys for the routes that run a model (`MATRICXON_API_KEYS`).

    A key comes as `Authorization: Bearer <key>` (what OpenAI SDKs send) or `X-API-Key: <key>`.
    No keys configured = open. Keys are compared as SHA-256 digests in constant time, so neither a
    key's length nor its first matching characters show in the answer time.
    """

    def __init__(self, keys_of: Callable[[], str]) -> None:
        self._keys_of = keys_of

    def _digests(self) -> list[bytes]:
        keys = [k.strip() for k in self._keys_of().split(",")]
        return [hashlib.sha256(k.encode()).digest() for k in keys if k]

    @property
    def enabled(self) -> bool:
        return bool(self._digests())

    @staticmethod
    def given_key(request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth[:7].lower() == "bearer ":
            return auth[7:].strip()
        return request.headers.get("X-API-Key", "").strip()

    def check(self, request: Request) -> None:
        digests = self._digests()
        if not digests:
            return
        given = hashlib.sha256(self.given_key(request).encode()).digest()
        # Every key is compared, so the time doesn't show which one (if any) matched.
        matched = [hmac.compare_digest(given, d) for d in digests]
        if not any(matched):
            raise HTTPException(
                status_code=401,
                detail="a valid API key is needed (Authorization: Bearer <key>)",
                headers={"WWW-Authenticate": "Bearer"},
            )


_auth = ApiKeyAuth(lambda: settings.api_keys)


def get_api_key_auth() -> ApiKeyAuth:
    return _auth


async def require_api_key(request: Request) -> None:
    """FastAPI dependency for the routes that run a model, and /metrics."""
    _auth.check(request)


async def require_admin_or_current_key(request: Request) -> None:
    """For changing the key: signed with the admin secret when one is set; otherwise the caller
    must send the current key, so nobody can simply remove it. Open only while neither is set."""
    from app.server.admin_auth import get_admin_auth

    admin = get_admin_auth()
    if admin.enabled:
        await admin.check(request)
    else:
        _auth.check(request)
