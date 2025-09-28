from typing import Any, Dict

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

