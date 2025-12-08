from typing import Any, Dict, List, Set

from .base import DeleteKey, Trunk


class PruneKeysTrunk(Trunk):
    """
    Example trunk demonstrating DeleteKey usage to drop keys from context.

    Params:
      - remove_keys: List[str] of top-level context keys to delete (if present).

    This trunk does not require any input keys and only deletes keys.
    """

    def __init__(self, remove_keys: List[str] | None = None, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._remove_keys = list(remove_keys or [])

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return set(self._remove_keys)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled or not self._remove_keys:
            return {}
        # Return DeleteKey for each requested key; AspenNet will drop them
        return {k: DeleteKey() for k in self._remove_keys}
