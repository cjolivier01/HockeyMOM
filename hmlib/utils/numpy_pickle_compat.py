from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator


def _install_numpy_core_aliases() -> list[str]:
    """Provide `numpy._core.*` aliases for NumPy<2.

    Some upstream checkpoints were saved with NumPy>=2 and reference internal
    modules under `numpy._core.*`. On NumPy<2 those modules live under
    `numpy.core.*`. We alias the relevant modules in `sys.modules` so
    `torch.load()` can unpickle them without requiring a NumPy upgrade.

    This is intentionally scoped and should only be enabled around checkpoint
    loading to avoid affecting other imports (e.g., OpenCV).
    """
    mapping: dict[str, ModuleType] = {}

    try:
        core = importlib.import_module("numpy.core")
    except Exception:
        return []

    mapping["numpy._core"] = core

    # Seed a small set of commonly referenced submodules.
    for name in (
        "numpy.core._multiarray_umath",
        "numpy.core.multiarray",
        "numpy.core.numeric",
        "numpy.core.fromnumeric",
        "numpy.core.umath",
        "numpy.core._ufunc_config",
    ):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        mapping[name.replace("numpy.core", "numpy._core", 1)] = mod

    inserted: list[str] = []
    for alias, mod in mapping.items():
        if alias not in sys.modules:
            sys.modules[alias] = mod
            inserted.append(alias)
    return inserted


@contextmanager
def numpy2_pickle_compat() -> Iterator[None]:
    """Context manager that enables NumPy>=2 pickle compatibility on NumPy<2."""
    inserted = _install_numpy_core_aliases()
    try:
        yield
    finally:
        for name in inserted:
            sys.modules.pop(name, None)
