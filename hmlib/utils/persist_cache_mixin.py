from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch


class PersistCacheMixin:
    """
    Add to your nn.Module to cache call-invariant tensors when `persist=True`.
    Invariants enforced by default: (H, W), dtype, device. You can include more via `extras`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._persist_cache: Optional[Dict[str, Any]] = None

    def clear_persist_cache(self) -> None:
        """Manually drop all cached tensors (e.g., after changing model config/device)."""
        self._persist_cache = None

    # ---- internal helpers ----
    def _persist_fingerprint(self, image: torch.Tensor, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fingerprint of things that must not change across persisted calls."""
        H, W = int(image.shape[-2]), int(image.shape[-1])  # ignore batch size
        fp = {
            "spatial": (H, W),
            "dtype": str(image.dtype),
            "device": str(image.device),
            # Anything else that would invalidate cached tensors (e.g., ksize/sigma/order/format/etc.)
            "extras": tuple(sorted((extras or {}).items())),
        }
        return fp

    def _persist_init_or_assert(
        self, persist: bool, image: torch.Tensor, extras: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the cache on first persisted call, or assert invariants thereafter."""
        if not persist:
            return
        fp = self._persist_fingerprint(image, extras)
        if self._persist_cache is None:
            self._persist_cache = {"fingerprint": fp, "tensors": {}}
        else:
            assert self._persist_cache["fingerprint"] == fp, (
                "Persistent cache invalidated: input size/type or relevant config changed.\n"
                f"Expected: {self._persist_cache['fingerprint']}\n"
                f"Got:      {fp}\n"
                "Call clear_persist_cache(), or pass persist=False for a one-off recompute."
            )

    def _persist_get(self, key: str, factory: Callable[[], torch.Tensor], persist: bool) -> torch.Tensor:
        """
        Return cached tensor if available (when persist=True); otherwise compute.
        Tensors are detached on cache write to avoid holding onto past graphs.
        """
        if not persist:
            return factory()

        assert self._persist_cache is not None, "Internal: persist cache not initialized."
        cache = self._persist_cache["tensors"]
        if key not in cache:
            cache[key] = factory().detach()
        return cache[key]
