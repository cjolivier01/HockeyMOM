from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class ProfilerOptions:
    enabled: bool = False
    save_dir: Optional[str] = None
    export_per_iter: bool = False
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    include_cuda: bool = False
    trace_basename: str = "trace"


class HmProfiler:
    """Lightweight wrapper around torch.profiler with zero overhead when disabled.

    - When disabled, all methods are no-ops and return cheap context managers.
    - When enabled, creates a torch.profiler.profile context and exposes rf(name)
      for record_function blocks and step() for per-iteration stepping.
    - Exports Perfetto/Chrome trace JSON on close (and optionally per iteration).
    """

    def __init__(self, opts: Optional[ProfilerOptions] = None):
        self._opts: ProfilerOptions = opts or ProfilerOptions(enabled=False)
        self._prof: Optional[torch.profiler.profile] = None
        self._step: int = 0
        self._closed: bool = False

        # Lazily ensure save dir exists if profiling is enabled
        if self._opts.enabled and self._opts.save_dir:
            try:
                os.makedirs(self._opts.save_dir, exist_ok=True)
            except Exception:
                pass

    # Cheap flag
    @property
    def enabled(self) -> bool:
        return bool(self._opts.enabled)

    # record_function wrapper
    def rf(self, name: str):
        if not self.enabled:
            return _NullContext()
        return torch.autograd.profiler.record_function(name)  # type: ignore[attr-defined]

    # NVTX convenience (safe no-op when disabled or no CUDA)
    def nvtx_range(self, name: str):
        if not (self.enabled and torch.cuda.is_available()):
            return _NullContext()
        try:
            return torch.cuda.nvtx.range(name)  # type: ignore[attr-defined]
        except Exception:
            return _NullContext()

    def __enter__(self):
        if not self.enabled:
            return self

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._opts.include_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Export handler
        def _on_trace_ready(prof: torch.profiler.profile):  # type: ignore
            if not self._opts.save_dir:
                return
            path = os.path.join(
                self._opts.save_dir, f"{self._opts.trace_basename}_{self._step:06d}.json"
            )
            try:
                prof.export_chrome_trace(path)
            except Exception:
                pass

        kwargs = dict(
            activities=activities,
            record_shapes=bool(self._opts.record_shapes),
            profile_memory=bool(self._opts.profile_memory),
            with_stack=bool(self._opts.with_stack),
        )
        if self._opts.export_per_iter and self._opts.save_dir:
            kwargs["on_trace_ready"] = _on_trace_ready  # type: ignore[assignment]

        self._prof = torch.profiler.profile(**kwargs)  # type: ignore[arg-type]
        self._prof.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self._prof is None:
            return False

        prof = self._prof
        self._prof = None

        try:
            prof.__exit__(exc_type, exc, tb)
        finally:
            self._closed = True

        if not self._opts.export_per_iter and self._opts.save_dir:
            # Export once after profiler context closes
            path = os.path.join(
                self._opts.save_dir, f"{self._opts.trace_basename}.json"
            )
            try:
                prof.export_chrome_trace(path)
            except Exception:
                pass

        return False

    def step(self):
        if not self.enabled or self._prof is None:
            return
        self._prof.step()
        self._step += 1


# Convenience factory from argparse-like args
def build_profiler_from_args(args, save_dir_fallback: Optional[str] = None) -> HmProfiler:
    enabled = bool(getattr(args, "profile", False) or getattr(args, "profiler", False))
    save_dir = getattr(args, "profile_dir", None) or save_dir_fallback
    opts = ProfilerOptions(
        enabled=enabled,
        save_dir=save_dir,
        export_per_iter=bool(getattr(args, "profile_export_per_iter", False)),
        record_shapes=bool(getattr(args, "profile_record_shapes", False)),
        profile_memory=bool(getattr(args, "profile_memory", False)),
        with_stack=bool(getattr(args, "profile_with_stack", False)),
        include_cuda=bool(getattr(args, "profile_gpu", False)),
        trace_basename=str(getattr(args, "profile_trace_basename", "trace")),
    )
    return HmProfiler(opts)


# Null context helper for call sites
def nullcontext():
    return _NullContext()
