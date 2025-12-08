"""Base interfaces and helpers for Aspen pipeline trunks."""

from contextlib import nullcontext
from typing import Any, Dict, Optional, Set

import torch


class Trunk(torch.nn.Module):
    """Base trunk interface. Subclasses implement forward(context) -> dict.

    Trunks read from and write to a shared 'context' dict. The return value
    should be a dict of new or updated context entries.
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self._profiler: Optional[Any] = None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        raise NotImplementedError

    # Keys this trunk wants to read from the context. If empty, receives full context.
    def input_keys(self) -> Set[str]:
        return set()

    # Keys this trunk promises to set/modify/delete.
    # If empty, AspenNet will treat all returned keys as modifications.
    def output_keys(self) -> Set[str]:
        return set()

    # Profiler plumbing -------------------------------------------------
    def set_profiler(self, profiler: Optional[Any]) -> None:
        """Inject a profiler instance provided by AspenNet."""
        self._profiler = profiler

    @property
    def profiler(self):
        """Return a context manager for ``with`` blocks or ``nullcontext``."""
        return self._profiler if self._profiler is not None else nullcontext()

    def profile_scope(self, label: str):
        """Return a record_function context or no-op when profiling is disabled."""
        profiler = self._profiler
        if profiler is None:
            return nullcontext()
        return profiler.rf(label)


class DeleteKey:
    """Sentinel indicating a key should be removed from context."""

    def __repr__(self) -> str:
        return "<DeleteKey>"
