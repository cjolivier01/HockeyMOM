from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Dict, Optional, Set

from .base import Plugin


def _import_symbol(path: str) -> Any:
    mod_name, _, attr = path.rpartition(".")
    if not mod_name:
        raise ValueError(f"Invalid import path: {path}")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


def _build_param(value: Any) -> Any:
    if isinstance(value, dict) and ("class" in value or "callable" in value):
        return _build_from_spec(value)
    if isinstance(value, dict):
        return {k: _build_param(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_build_param(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_build_param(v) for v in value)
    return value


def _build_from_spec(spec: Dict[str, Any]) -> Any:
    target_path = spec.get("class") or spec.get("callable")
    if not target_path:
        raise ValueError("Factory spec requires 'class' or 'callable'.")
    target = _import_symbol(str(target_path))
    args = spec.get("args", []) or []
    params = spec.get("params", {}) or {}
    built_args = [_build_param(v) for v in args]
    built_params = {k: _build_param(v) for k, v in params.items()}
    return target(*built_args, **built_params)


def _ctx_value(context: Dict[str, Any], key: Optional[str]) -> Optional[Any]:
    if not key:
        return None
    if key in context:
        return context[key]
    shared = context.get("shared")
    if isinstance(shared, dict):
        return shared.get(key)
    return None


class DataLoaderPlugin(Plugin):
    """
    Pulls the next batch from a DataLoader/iterator and merges it into context.

    Params:
      - dataloader: DataLoader/iterable instance or factory spec dict.
      - dataloader_key: Context/shared key to look up a dataloader (default "dataloader").
      - iterator_key: Context/shared key to look up an iterator (default "dataloader_iter").
      - wrap_key: If set, output {wrap_key: batch} instead of merging batch mapping.
      - batch_key: Fallback key for non-mapping batches when wrap_key is unset.
      - strict_dict: Require batch to be a Mapping when wrap_key is unset (default True).
      - repeat: Restart the iterator when exhausted (default False).
      - batch_index_key: Optional key for a monotonically increasing batch index.
      - start_index: Initial value for batch_index_key.
      - return_dataloader: Include the dataloader in outputs (default False).
      - return_iterator: Include the iterator in outputs (default False).

    Returns:
      - Batch mapping (or wrapped) plus optional batch index/dataloader/iterator.
    """

    def __init__(
        self,
        dataloader: Any = None,
        dataloader_key: str = "dataloader",
        iterator_key: str = "dataloader_iter",
        wrap_key: Optional[str] = None,
        batch_key: str = "batch",
        strict_dict: bool = True,
        repeat: bool = False,
        batch_index_key: Optional[str] = None,
        start_index: int = 0,
        return_dataloader: bool = False,
        return_iterator: bool = False,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._dataloader_key = dataloader_key or None
        self._iterator_key = iterator_key or None
        self._wrap_key = wrap_key or None
        self._batch_key = batch_key or "batch"
        self._strict_dict = bool(strict_dict)
        self._repeat = bool(repeat)
        self._batch_index_key = batch_index_key or None
        self._start_index = int(start_index or 0)
        self._next_batch_index = self._start_index
        self._return_dataloader = bool(return_dataloader)
        self._return_iterator = bool(return_iterator)

        self._dataloader_spec: Optional[Dict[str, Any]] = None
        self._dataloader: Optional[Any] = None
        self._iterator: Optional[Any] = None

        if isinstance(dataloader, dict):
            self._dataloader_spec = dataloader
        elif dataloader is not None:
            self._dataloader = dataloader

    def _set_dataloader(self, dataloader: Any) -> None:
        if dataloader is self._dataloader:
            return
        self._dataloader = dataloader
        self._iterator = None
        self._next_batch_index = self._start_index

    def _set_iterator(self, iterator: Any) -> None:
        if iterator is self._iterator:
            return
        self._iterator = iterator
        self._next_batch_index = self._start_index

    def _ensure_dataloader(self, context: Dict[str, Any]) -> Any:
        ctx_dl = _ctx_value(context, self._dataloader_key)
        if ctx_dl is not None:
            self._set_dataloader(ctx_dl)
        elif self._dataloader is None and self._dataloader_spec is not None:
            self._set_dataloader(_build_from_spec(self._dataloader_spec))
        if self._dataloader is None:
            raise RuntimeError("DataLoaderPlugin requires a dataloader or dataloader spec.")
        return self._dataloader

    def _ensure_iterator(self, context: Dict[str, Any], dataloader: Any) -> Any:
        ctx_iter = _ctx_value(context, self._iterator_key)
        if ctx_iter is not None:
            self._set_iterator(ctx_iter)
            return ctx_iter
        if self._iterator is None:
            self._iterator = iter(dataloader)
        return self._iterator

    def _format_batch(self, batch: Any) -> Dict[str, Any]:
        if self._wrap_key:
            return {self._wrap_key: batch}
        if isinstance(batch, Mapping):
            return dict(batch)
        if self._strict_dict:
            raise TypeError(
                f"Expected dataloader batch to be a mapping, got {type(batch).__name__}"
            )
        return {self._batch_key: batch}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        dataloader = self._ensure_dataloader(context)
        iterator = self._ensure_iterator(context, dataloader)

        try:
            batch = next(iterator)
        except StopIteration:
            if not self._repeat:
                raise
            iterator = iter(dataloader)
            self._iterator = iterator
            batch = next(iterator)

        out = self._format_batch(batch)
        if self._batch_index_key:
            out[self._batch_index_key] = self._next_batch_index
            self._next_batch_index += 1
        if self._return_dataloader and self._dataloader_key:
            out[self._dataloader_key] = dataloader
        if self._return_iterator and self._iterator_key:
            out[self._iterator_key] = iterator
        return out

    def input_keys(self) -> Set[str]:
        keys: Set[str] = set()
        if self._dataloader_key:
            keys.add(self._dataloader_key)
        if self._iterator_key:
            keys.add(self._iterator_key)
        return keys

    def output_keys(self) -> Set[str]:
        if not self._wrap_key:
            return set()
        keys: Set[str] = {self._wrap_key}
        if self._batch_index_key:
            keys.add(self._batch_index_key)
        if self._return_dataloader and self._dataloader_key:
            keys.add(self._dataloader_key)
        if self._return_iterator and self._iterator_key:
            keys.add(self._iterator_key)
        return keys
