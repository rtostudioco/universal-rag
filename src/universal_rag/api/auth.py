"""Autenticación opcional por API key (Fase 4).

Si ``RAG_API_KEY`` está definida, todos los endpoints protegidos exigen el
header ``X-API-Key`` o ``Authorization: Bearer <key>``. ``/`` y ``/health``
permanecen abiertos para monitoreo.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from universal_rag.core.config import Settings

_UNAUTHORIZED_DETAIL = "API key ausente o inválida. Usa el header 'X-API-Key' o 'Authorization: Bearer <key>'."


def create_auth_dependency(settings: Settings):
    """Fábrica de la dependency de FastAPI según la config actual."""
    expected = settings.api_key or ""

    async def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not expected:  # auth desactivada
            return
        provided = x_api_key
        if not provided and authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_UNAUTHORIZED_DETAIL,
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_api_key
