"""WebSocket trace streaming for live dashboards.

Clients subscribe to ``/ws/v1/stream`` and receive JSON messages every
time the runtime they're attached to emits a TraceEvent. Each robot's
runtime can register an emitter; the StreamManager fans events out to
every connected websocket subscriber.

Conditional — only available when fastapi is installed (same dep as
the dashboard module). The runtime side has zero new deps; emitting an
event into a StreamManager just calls ``manager.publish()``.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamManager:
    """Fan-out websocket pubsub. Synchronous publish, async subscribe.

    Storage is a bounded ring buffer per robot so newly-connected
    clients can receive a small replay of the most recent events
    (so dashboards don't open completely blank).
    """

    max_history: int = 64
    _buffers: dict[str, deque[dict[str, Any]]] = field(default_factory=dict)
    _subscribers: list = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def publish(self, robot_name: str, event: dict[str, Any]) -> None:
        """Synchronous: append to buffer, notify async subscribers (best-effort)."""
        buf = self._buffers.setdefault(robot_name, deque(maxlen=self.max_history))
        envelope = {"robot": robot_name, **event}
        buf.append(envelope)
        # Notify subscribers via their queue.put_nowait (non-blocking).
        for q in list(self._subscribers):
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                # Subscriber's queue is saturated; drop this event for them
                # rather than blocking the publisher.
                continue

    def history(self, robot_name: str) -> list[dict[str, Any]]:
        return list(self._buffers.get(robot_name, ()))

    async def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: "asyncio.Queue[dict[str, Any]]") -> None:
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


def attach_streaming(app, manager: StreamManager) -> None:
    """Register the /ws/v1/stream WebSocket endpoint on a FastAPI app.

    Raises ImportError if fastapi isn't installed (no-op at install
    time — only the call site fails, which keeps the runtime a no-fastapi
    dependency).
    """
    try:
        from fastapi import WebSocket, WebSocketDisconnect  # type: ignore
    except ImportError as e:
        raise ImportError(
            "WebSocket streaming requires fastapi. pip install ghostloop[dashboard]"
        ) from e

    @app.websocket("/ws/v1/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        # Send any backlog from each robot so the client doesn't get a
        # blank screen on connect.
        for robot_name, buf in manager._buffers.items():
            for envelope in list(buf):
                await ws.send_text(json.dumps(envelope))
        q = await manager.subscribe()
        try:
            while True:
                envelope = await q.get()
                await ws.send_text(json.dumps(envelope))
        except WebSocketDisconnect:
            pass
        finally:
            await manager.unsubscribe(q)
