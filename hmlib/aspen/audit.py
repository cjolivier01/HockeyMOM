from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.image import make_channels_last, make_visible_image

try:  # pragma: no cover - optional dependency (used only when dumping PNGs)
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


_JSONDict = Dict[str, Any]


def _sanitize_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    safe = safe.strip("._")
    return safe or "item"


def _iter_path_tokens(path: str) -> List[Union[str, int]]:
    tokens: List[Union[str, int]] = []
    for raw in str(path).split("."):
        raw = raw.strip()
        if not raw:
            continue
        if raw.isdigit():
            tokens.append(int(raw))
        else:
            tokens.append(raw)
    return tokens


def _get_by_path(root: Any, path: str) -> Any:
    cur = root
    for token in _iter_path_tokens(path):
        if cur is None:
            return None
        if isinstance(token, int):
            if isinstance(cur, (list, tuple)) and 0 <= token < len(cur):
                cur = cur[token]
            else:
                return None
            continue
        if isinstance(cur, Mapping):
            cur = cur.get(token)
            continue
        # Best-effort attribute access (e.g., namespaces / dataclasses)
        if hasattr(cur, token):
            cur = getattr(cur, token)
            continue
        return None
    return cur


def _maybe_to_cpu_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert a tensor/array-like to a CPU numpy array (blocking)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, StreamTensorBase):
        value = value.get()
    value = unwrap_tensor(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            t = value.detach()
            if t.is_cuda:
                t = t.cpu()
            return np.ascontiguousarray(t.numpy())
    except Exception:
        # torch not available or failed conversion
        return None
    return None


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint_array(arr: np.ndarray) -> Tuple[str, _JSONDict]:
    arr_c = np.ascontiguousarray(arr)
    h = _hash_bytes(arr_c.tobytes())
    meta: _JSONDict = {
        "shape": list(arr_c.shape),
        "dtype": str(arr_c.dtype),
    }
    try:
        if arr_c.size:
            # Use float64 for stable JSON serialization.
            meta["min"] = float(arr_c.min())
            meta["max"] = float(arr_c.max())
    except Exception:
        pass
    return h, meta


def _is_image_like(arr: np.ndarray) -> bool:
    if arr.dtype != np.uint8:
        return False
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return True
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return True
    return False


def _to_bgr_hwc_uint8(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr.dtype != np.uint8:
        return None
    a = arr
    if a.ndim != 3:
        return None
    if a.shape[-1] in (1, 3, 4):
        return np.ascontiguousarray(a)
    if a.shape[0] in (1, 3, 4):
        # CHW -> HWC
        return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))
    return None


def _extract_frame_ids(ctx: Mapping[str, Any]) -> Optional[List[int]]:
    """Extract frame ids from common context keys (best-effort)."""
    for key in ("frame_ids", "ids"):
        if key not in ctx:
            continue
        ids_val = ctx.get(key)
        if ids_val is None:
            continue
        if isinstance(ids_val, StreamTensorBase):
            try:
                ids_val = ids_val.get()
            except Exception:
                ids_val = unwrap_tensor(ids_val, get=True)
        if isinstance(ids_val, (list, tuple)):
            try:
                return [int(x) for x in ids_val]
            except Exception:
                continue
        try:
            import torch

            if isinstance(ids_val, torch.Tensor):
                ids_cpu = ids_val.detach().cpu().reshape(-1).tolist()
                return [int(x) for x in ids_cpu]
        except Exception:
            continue
    frame_id = ctx.get("frame_id")
    if frame_id is not None:
        try:
            return [int(frame_id)]
        except Exception:
            return None
    stitch_inputs = ctx.get("stitch_inputs")
    if stitch_inputs is not None:
        left = None
        if isinstance(stitch_inputs, dict):
            left = stitch_inputs.get("left")
        elif isinstance(stitch_inputs, (list, tuple)) and stitch_inputs:
            left = stitch_inputs[0]
        if isinstance(left, dict):
            for key in ("frame_ids", "ids", "img_id"):
                ids_val = left.get(key)
                if ids_val is None:
                    continue
                if isinstance(ids_val, StreamTensorBase):
                    try:
                        ids_val = ids_val.get()
                    except Exception:
                        ids_val = unwrap_tensor(ids_val, get=True)
                if isinstance(ids_val, (list, tuple)):
                    try:
                        return [int(x) for x in ids_val]
                    except Exception:
                        continue
                try:
                    import torch

                    if isinstance(ids_val, torch.Tensor):
                        ids_cpu = ids_val.detach().cpu().reshape(-1).tolist()
                        return [int(x) for x in ids_cpu]
                except Exception:
                    continue
    return None


@dataclass(frozen=True)
class AspenAuditConfig:
    out_dir: Path
    reference_dir: Optional[Path] = None
    plugins: Optional[Sequence[str]] = None
    dump_images: bool = False
    fail_fast: bool = True


class AspenAuditHook:
    """Capture and optionally compare per-plugin image fingerprints.

    This is designed to pinpoint missing CUDA stream synchronization by
    recording exact (sha256) hashes of intermediate frame tensors.
    """

    def __init__(self, config: AspenAuditConfig) -> None:
        self._cfg = config
        self._lock = threading.Lock()
        self._out_dir = Path(config.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._audit_path = self._out_dir / "audit.jsonl"
        self._mismatch_path = self._out_dir / "mismatches.jsonl"
        self._plugins = set(config.plugins) if config.plugins else None
        self._dump_images = bool(config.dump_images)
        self._fail_fast = bool(config.fail_fast)

        self._ref: Dict[Tuple[int, str, str, str], str] = {}
        ref_dir = config.reference_dir
        if ref_dir is not None:
            ref_path = Path(ref_dir) / "audit.jsonl"
            if ref_path.is_file():
                self._ref = self._load_reference(ref_path)
            else:
                raise FileNotFoundError(f"Reference audit file not found: {ref_path}")

    def _load_reference(self, path: Path) -> Dict[Tuple[int, str, str, str], str]:
        ref: Dict[Tuple[int, str, str, str], str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                try:
                    frame_id = int(rec["frame_id"])
                    plugin = str(rec["plugin"])
                    phase = str(rec["phase"])
                    key = str(rec["key"])
                    h = str(rec["hash"])
                except Exception:
                    continue
                ref[(frame_id, plugin, phase, key)] = h
        logger.info("AspenAuditHook loaded %d reference records from %s", len(ref), path)
        return ref

    def close(self) -> None:
        # Currently no long-lived resources beyond files opened per-write.
        return None

    def before_plugin(
        self,
        plugin_name: str,
        subctx: Mapping[str, Any],
        full_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._plugins is not None and plugin_name not in self._plugins:
            return
        self._capture(plugin_name, "before", subctx, ctx_for_ids=full_context or subctx)

    def after_plugin(
        self,
        plugin_name: str,
        subctx: Mapping[str, Any],
        out: Mapping[str, Any],
        full_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._plugins is not None and plugin_name not in self._plugins:
            return
        self._capture(plugin_name, "after", out, ctx_for_ids=full_context or subctx)

    def _capture(
        self,
        plugin_name: str,
        phase: str,
        data: Mapping[str, Any],
        *,
        ctx_for_ids: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Best-effort batch-aware frame ids
        ids_ctx = ctx_for_ids if ctx_for_ids is not None else data
        frame_ids = _extract_frame_ids(ids_ctx) or []

        # Capture spec: mostly images, plus stitching pre-inputs.
        paths: List[str] = []
        if phase == "before":
            paths.extend(["img", "original_images", "end_zone_img"])
            if plugin_name == "stitching":
                # Prefer left/right from stitch_inputs when present (list or dict).
                paths.extend(["stitch_inputs.0.img", "stitch_inputs.1.img"])
                paths.extend(["stitch_inputs.left.img", "stitch_inputs.right.img"])
        else:
            paths.extend(["img", "original_images", "end_zone_img"])

        records: List[_JSONDict] = []
        mismatches: List[_JSONDict] = []

        for path in paths:
            value = _get_by_path(data, path)
            arr = _maybe_to_cpu_numpy(value)
            if arr is None:
                continue

            # If batched, fingerprint each element so frame_id mapping is meaningful.
            if arr.ndim == 4:
                batch = int(arr.shape[0])
                for i in range(batch):
                    frame_id = frame_ids[i] if i < len(frame_ids) else None
                    rec, mismatch = self._record_one(
                        plugin_name=plugin_name,
                        phase=phase,
                        key=path,
                        frame_id=frame_id,
                        frame_index=i,
                        arr=arr[i],
                    )
                    records.append(rec)
                    if mismatch is not None:
                        mismatches.append(mismatch)
            else:
                frame_id = frame_ids[0] if frame_ids else None
                rec, mismatch = self._record_one(
                    plugin_name=plugin_name,
                    phase=phase,
                    key=path,
                    frame_id=frame_id,
                    frame_index=None,
                    arr=arr,
                )
                records.append(rec)
                if mismatch is not None:
                    mismatches.append(mismatch)

        if not records and not mismatches:
            return

        with self._lock:
            with self._audit_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

            if mismatches:
                with self._mismatch_path.open("a", encoding="utf-8") as f:
                    for mm in mismatches:
                        f.write(json.dumps(mm, sort_keys=True) + "\n")
                if self._fail_fast:
                    raise RuntimeError(
                        f"AspenAuditHook detected {len(mismatches)} mismatch(es); "
                        f"see {self._mismatch_path}"
                    )

    def _record_one(
        self,
        *,
        plugin_name: str,
        phase: str,
        key: str,
        frame_id: Optional[int],
        frame_index: Optional[int],
        arr: np.ndarray,
    ) -> Tuple[_JSONDict, Optional[_JSONDict]]:
        h, meta = _fingerprint_array(arr)
        rec: _JSONDict = {
            "frame_id": int(frame_id) if frame_id is not None else None,
            "frame_index": int(frame_index) if frame_index is not None else None,
            "plugin": plugin_name,
            "phase": phase,
            "key": key,
            "hash": h,
            **meta,
        }

        mismatch: Optional[_JSONDict] = None
        if self._ref and frame_id is not None:
            ref_key = (int(frame_id), plugin_name, phase, key)
            expected = self._ref.get(ref_key)
            if expected is None or expected != h:
                mismatch = {
                    "frame_id": int(frame_id),
                    "frame_index": int(frame_index) if frame_index is not None else None,
                    "plugin": plugin_name,
                    "phase": phase,
                    "key": key,
                    "expected": expected,
                    "actual": h,
                }
                self._dump_images_on_mismatch(
                    plugin_name=plugin_name,
                    phase=phase,
                    key=key,
                    frame_id=int(frame_id),
                    arr=arr,
                )
        if self._dump_images and frame_id is not None:
            self._dump_image(
                plugin_name=plugin_name,
                phase=phase,
                key=key,
                frame_id=frame_id,
                arr=arr,
            )
        return rec, mismatch

    def _dump_images_on_mismatch(
        self, *, plugin_name: str, phase: str, key: str, frame_id: int, arr: np.ndarray
    ) -> None:
        # Always dump the current image on mismatch (even if dump_images=False).
        try:
            self._dump_image(
                plugin_name=plugin_name,
                phase=phase,
                key=key,
                frame_id=frame_id,
                arr=arr,
            )
        except Exception:
            logger.debug("AspenAuditHook failed to dump mismatch image", exc_info=True)

    def _dump_image(
        self, *, plugin_name: str, phase: str, key: str, frame_id: int, arr: np.ndarray
    ) -> None:
        if cv2 is None:
            return
        if not _is_image_like(arr):
            return
        bgr = _to_bgr_hwc_uint8(arr)
        if bgr is None:
            return
        # Make it visible-friendly (handles grayscale/alpha, etc.).
        try:
            import torch

            t = torch.from_numpy(bgr)
            t = make_channels_last(t)
            vis = make_visible_image(t, force_numpy=True)
            if isinstance(vis, np.ndarray):
                bgr = np.ascontiguousarray(vis)
        except Exception:
            pass

        out_dir = self._out_dir / "images" / _sanitize_component(phase) / _sanitize_component(plugin_name)
        out_dir = out_dir / _sanitize_component(key.replace(".", "_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{int(frame_id):06d}.png"
        cv2.imwrite(str(out_path), bgr)
