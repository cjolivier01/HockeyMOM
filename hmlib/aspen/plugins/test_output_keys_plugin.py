from typing import Any, Dict, Set

from .base import Plugin


class BadOutputPlugin(Plugin):
    """Plugin used in tests that always returns an undeclared key."""

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return {"allowed_key"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        return {"allowed_key": 1, "extra_key": 2}


class FlakyOutputPlugin(Plugin):
    """Plugin used in tests that returns an extra key after the first call."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return {"value"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            return {"value": self.calls}
        return {"value": self.calls, "extra_key": 99}
