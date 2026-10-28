"""Bound request bodies before FastAPI parses JSON payloads."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import JSONResponse


DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024


class RequestBodyLimitMiddleware:
    """Buffer at most ``max_body_bytes`` before handing a request downstream.

    Content-Length is used only for an early rejection. Requests without that
    header, or with a misleading smaller value, are checked while receiving
    each ASGI body chunk. The server may already have allocated an individual
    chunk before application middleware sees it.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], *, max_body_bytes: int) -> None:
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[dict[str, Any]]], send: Callable[..., Awaitable[None]]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await _send_too_large(send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_body_bytes:
                await _send_too_large(send)
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _content_length(scope: dict[str, Any]) -> int | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed >= 0 else None
    return None


async def _send_too_large(send: Callable[..., Awaitable[None]]) -> None:
    response = JSONResponse(
        status_code=413,
        content={"detail": "Request body exceeds the configured size limit."},
    )
    await response({"type": "http", "asgi": {"version": "3.0"}}, _empty_receive, send)


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.disconnect"}
