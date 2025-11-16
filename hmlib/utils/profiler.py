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
    # Optional step-gating: enable profiling only for a window of iterations.
    # These are expressed in zero-based iteration indices internally.
    start_step_idx: Optional[int] = None
    step_count: Optional[int] = None


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
        # _step counts iterations while the profiler is active (for trace naming)
        self._step: int = 0
        # _global_step counts all logical iterations (for step-gating)
        self._global_step: int = 0
        # Whether the profiling window is currently active
        self._active: bool = False
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
        # "Enabled" reflects whether profiling was requested; actual recording
        # may still be gated on step indices.
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
            with_cuda=bool(self._opts.include_cuda),
        )
        if self._opts.export_per_iter and self._opts.save_dir:
            kwargs["on_trace_ready"] = _on_trace_ready  # type: ignore[assignment]

        self._prof = torch.profiler.profile(**kwargs)  # type: ignore[arg-type]
        self._prof.__enter__()
        self._active = True
        self._step = 0
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

        self._active = False
        return False

    def step(self):
        """
        Advance the logical iteration counter and, if configured, drive the
        underlying torch.profiler window (start/step/stop).
        """
        if not self.enabled:
            return

        # Global iteration index (used for gating)
        self._global_step += 1

        # Resolve gating configuration (convert None -> "always on")
        start_idx = self._opts.start_step_idx
        step_count = self._opts.step_count

        # If no explicit start index is configured, start immediately on first step()
        if start_idx is None:
            start_idx = 0

        # If profiling window hasn't started yet and we've reached the start index,
        # lazily enter the torch.profiler context.
        if not self._active and not self._closed and self._global_step - 1 >= start_idx:
            # Avoid re-entering if already active/closed
            self.__enter__()

        # If we're not currently active or profiler failed to initialize, nothing to do.
        if not self._active or self._prof is None:
            return

        # Step the underlying profiler and bump active-step counter.
        self._prof.step()
        self._step += 1

        # If a finite window is configured and we've consumed it, close the profiler.
        if step_count is not None and step_count > 0 and self._step >= step_count:
            self.__exit__(None, None, None)


# Convenience factory from argparse-like args
def build_profiler_from_args(args, save_dir_fallback: Optional[str] = None) -> HmProfiler:
    enabled = bool(getattr(args, "profile", False) or getattr(args, "profiler", False))
    save_dir = getattr(args, "profile_dir", None) or save_dir_fallback
    profile_with_stack = getattr(args, "profile_with_stack", None)
    if profile_with_stack is None:
        profile_with_stack = bool(enabled)

    # Step-gated profiling: map 1-based CLI step to 0-based internal index.
    raw_start_step = getattr(args, "profile_step", None)
    start_step_idx = None
    if raw_start_step is not None:
        try:
            # Clamp to zero or above; interpret as 1-based from CLI.
            start_step_idx = max(int(raw_start_step) - 1, 0)
        except Exception:
            start_step_idx = 0

    raw_step_count = getattr(args, "profile_step_count", None)
    step_count = None
    if raw_step_count is not None:
        try:
            step_count = max(int(raw_step_count), 1)
        except Exception:
            step_count = 1

    opts = ProfilerOptions(
        enabled=enabled,
        save_dir=save_dir,
        export_per_iter=bool(getattr(args, "profile_export_per_iter", False)),
        record_shapes=bool(getattr(args, "profile_record_shapes", False)),
        profile_memory=bool(getattr(args, "profile_memory", False)),
        with_stack=bool(profile_with_stack),
        include_cuda=True,
        trace_basename=str(getattr(args, "profile_trace_basename", "trace")),
        start_step_idx=start_step_idx,
        step_count=step_count,
    )
    return HmProfiler(opts)


# Null context helper for call sites
def nullcontext():
    return _NullContext()
