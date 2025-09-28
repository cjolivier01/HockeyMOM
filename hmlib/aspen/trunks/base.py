from typing import Any, Dict, Iterable, Set

import torch


class Trunk(torch.nn.Module):
    """Base trunk interface. Subclasses implement forward(context) -> dict.

    Trunks read from and write to a shared 'context' dict. The return value
    should be a dict of new or updated context entries.
    """

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

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


class DeleteKey:
    """Sentinel indicating a key should be removed from context."""

    def __repr__(self) -> str:
        return "<DeleteKey>"

