from __future__ import annotations

import threading
import time
from typing import Any, Dict

from .base import Plugin


def _increment_done(shared: Dict[str, Any]) -> None:
    done_event = shared.get("done_event")
    if done_event is None:
        return
    shared["done_count"] = shared.get("done_count", 0) + 1
    expected = int(shared.get("expected") or 0)
    if expected and shared["done_count"] >= expected:
        done_event.set()


class RecordPlugin(Plugin):
    """Test plugin that records execution order into shared state."""

    def __init__(self, name: str, delay: float = 0.0, mark_done: bool = False):
        super().__init__()
        self._name = name
        self._delay = float(delay or 0.0)
        self._mark_done = bool(mark_done)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        shared = context.get("shared", {})
        log = shared.get("log")
        lock = shared.get("lock")
        seq = context.get("seq")

        if log is not None:
            if lock is not None:
                with lock:
                    log.setdefault(self._name, []).append(seq)
                    if self._mark_done:
                        _increment_done(shared)
            else:
                log.setdefault(self._name, []).append(seq)
                if self._mark_done:
                    _increment_done(shared)

        if self._delay:
            time.sleep(self._delay)
        return {}


class BarrierPlugin(Plugin):
    """Test plugin that waits on a shared barrier to confirm concurrency."""

    def __init__(self, name: str, mark_done: bool = False):
        super().__init__()
        self._name = name
        self._mark_done = bool(mark_done)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        shared = context.get("shared", {})
        barrier = shared.get("barrier")
        if barrier is None:
            raise RuntimeError("BarrierPlugin requires shared['barrier']")

        log = shared.get("log")
        lock = shared.get("lock")
        seq = context.get("seq")
        if log is not None:
            if lock is not None:
                with lock:
                    log.setdefault(self._name, []).append(seq)
            else:
                log.setdefault(self._name, []).append(seq)

        try:
            barrier.wait(timeout=float(shared.get("barrier_timeout", 2.0)))
        except threading.BrokenBarrierError as exc:
            raise RuntimeError("BarrierPlugin timed out waiting for sibling") from exc

        if self._mark_done:
            if lock is not None:
                with lock:
                    _increment_done(shared)
            else:
                _increment_done(shared)
        return {}


class KeyPlugin(Plugin):
    """Test plugin that returns a fixed mapping (for join/merge tests)."""

    def __init__(self, outputs: Dict[str, Any]):
        super().__init__()
        self._outputs = dict(outputs)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        return dict(self._outputs)

    def input_keys(self) -> set[str]:
        return set()

    def output_keys(self) -> set[str]:
        return set(self._outputs.keys())
