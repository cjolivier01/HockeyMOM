from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from .base import Plugin


class ActionRecognizerFactoryPlugin(Plugin):
    """
    Builds and caches an MMAction2 recognizer and exposes it via context.

    Params:
      - action_config: str path to MMAction2 config (skeleton model recommended)
      - action_checkpoint: Optional[str] checkpoint path/URL
      - device: Optional[str] device string (overrides context['device'] if provided)
      - label_map_path: Optional[str] path to a label map text file (one label per line)

    Provides in context:
      - action_recognizer: initialized mmaction model (eval mode)
      - action_label_map: Optional[List[str]] if label_map_path present
    """

    def __init__(
        self,
        action_config: Optional[str] = None,
        action_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        label_map_path: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._action_config = action_config
        self._action_checkpoint = action_checkpoint
        self._device = device
        self._label_map_path = label_map_path
        self._recognizer = None
        self._label_map: Optional[List[str]] = None

    @staticmethod
    def _ensure_mmaction_imported():
        try:
            import mmaction  # noqa: F401

            return
        except Exception:
            pass
        # Fallback: add vendored OpenMMLab path if present
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        vendored = os.path.join(root, "openmm", "mmaction2")
        if os.path.isdir(vendored) and vendored not in sys.path:
            sys.path.insert(0, vendored)
        import mmaction  # type: ignore  # noqa: F401

    def _load_label_map(self, path: Optional[str]) -> Optional[List[str]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            return [ln for ln in lines if ln]
        except Exception:
            return None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        if self._recognizer is None:
            self._ensure_mmaction_imported()
            from mmaction.apis import init_recognizer

            cfg = self._action_config
            ckpt = self._action_checkpoint
            dev = self._device
            if dev is None:
                device_obj = context.get("device")
                if device_obj is not None:
                    dev = str(device_obj)
            # Default to CUDA:0 if device remains None
            dev = dev or "cuda:0"
            self._recognizer = init_recognizer(cfg, ckpt, dev)
            # Try load label map, with a sensible default if not provided
            label_map = self._load_label_map(self._label_map_path)
            if label_map is None:
                # Default to NTU60 label list from vendored repo if present
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                ntu60_map = os.path.join(
                    root, "openmm", "mmaction2", "tools", "data", "skeleton", "label_map_ntu60.txt"
                )
                label_map = self._load_label_map(ntu60_map)
            self._label_map = label_map

        context["action_recognizer"] = self._recognizer
        if self._label_map is not None:
            context["action_label_map"] = self._label_map
        return {"action_recognizer": self._recognizer, "action_label_map": self._label_map}

    def input_keys(self):
        return {"device"}

    def output_keys(self):
        return {"action_recognizer", "action_label_map"}
