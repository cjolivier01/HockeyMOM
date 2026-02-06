from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch

from hmlib.builder import HM
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase

from .base import Plugin


@HM.register_module()
class RgbStatsCheckPlugin(Plugin):
    """
    Debug plugin that recomputes per-frame RGB statistics and compares them
    against reference stats captured earlier in the pipeline.

    Typical usage:
      - StitchDataset/StitchingPlugin compute 'left', 'right', 'stitched' stats and store
        them under context['debug_rgb_stats']['stitch'].
      - This plugin recomputes stats on a chosen image tensor (e.g.,
        'original_images') and logs a warning if they differ beyond a
        configurable tolerance.

    Results are attached under context['debug_rgb_stats_checks'] so downstream
    consumers can inspect them if needed.
    """

    def __init__(
        self,
        enabled: bool = True,
        source: str = "stitch",
        stats_key: str = "stitched",
        tensor_key: str = "original_images",
        atol: float = 1e-3,
        rtol: float = 1e-5,
        log_on_change: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._source = str(source)
        self._stats_key = str(stats_key)
        self._tensor_key = str(tensor_key)
        self._atol = float(atol)
        self._rtol = float(rtol)
        self._log_on_change = bool(log_on_change)

    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        if not self.enabled:
            return {}

        debug_stats: Dict[str, Any] = context.get("debug_rgb_stats", {}) or {}
        src_stats: Optional[Dict[str, Any]] = debug_stats.get(self._source)
        if not isinstance(src_stats, dict):
            return {}
        reference = src_stats.get(self._stats_key)
        if reference is None:
            return {}

        img = context.get(self._tensor_key)
        if img is None:
            return {}
        if isinstance(img, StreamTensorBase):
            img = img.get()
        if not isinstance(img, torch.Tensor):
            return {}

        try:
            changed, unchanged = MOTLoadVideoWithOrig.check_rgb_stats(
                img=img,
                reference=reference,
                atol=self._atol,
                rtol=self._rtol,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RgbStatsCheckPlugin failed to compare stats: %s", exc)
            return {}

        if changed and self._log_on_change:
            logger.warning(
                "RgbStatsCheckPlugin: RGB stats changed for %s.%s (tensor_key=%s)",
                self._source,
                self._stats_key,
                self._tensor_key,
            )

        # Attach results for downstream inspection.
        checks = context.get("debug_rgb_stats_checks")
        if not isinstance(checks, dict):
            checks = {}
        key = f"{self._source}.{self._stats_key}"
        checks[key] = {
            "tensor_key": self._tensor_key,
            "changed": bool(changed),
            "unchanged": bool(unchanged),
        }
        return {"debug_rgb_stats_checks": checks}

    def input_keys(self) -> Set[str]:
        return {"debug_rgb_stats", "debug_rgb_stats_checks", self._tensor_key}

    def output_keys(self) -> Set[str]:
        return {"debug_rgb_stats_checks"}
