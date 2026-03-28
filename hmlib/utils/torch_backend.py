"""Helpers for distinguishing CUDA, ROCm, and CPU torch runtimes."""

from __future__ import annotations

import torch


def torch_backend() -> str:
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def is_rocm_backend() -> bool:
    return torch_backend() == "rocm"


__all__ = ["is_rocm_backend", "torch_backend"]
