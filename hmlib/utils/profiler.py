from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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

    @see @ref build_profiler_from_args "build_profiler_from_args" for CLI integration.
    @see @ref hmlib.hm_opts.hm_opts "hm_opts" for the corresponding CLI flags.
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
                # total_busy, concurrent, kernel_with_memcpy = compute_cuda_kernel_times(path)
                # print(f"{total_busy=}, {concurrent=}, {kernel_with_memcpy=}")
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
            path = os.path.join(self._opts.save_dir, f"{self._opts.trace_basename}.json")
            try:
                prof.export_chrome_trace(path)
                # total_busy, concurrent, kernel_with_memcpy = compute_cuda_kernel_times(path)
                # print(f"{total_busy=}, {concurrent=}, {kernel_with_memcpy=}")
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
    """Construct an :class:`HmProfiler` from argparse-style CLI options.

    @param args: Namespace-like object with ``profile*`` fields from :mod:`hmlib.hm_opts`.
    @param save_dir_fallback: Directory used when ``args.profile_dir`` is empty.
    @return: Configured :class:`HmProfiler` instance.
    @see @ref HmProfiler "HmProfiler" for the profiling API.
    """
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

    step_count = getattr(args, "profile_step_count", 0)

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


TracePath = Union[str, Path]


def _load_trace_events(path: TracePath) -> Iterable[Dict]:
    """Load ``traceEvents`` list from a Chrome/Perfetto trace JSON file."""
    with open(Path(path), "r") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def _collect_kernel_intervals(
    events: Iterable[Dict],
) -> List[Tuple[float, float, Tuple[Optional[int], Optional[int]]]]:
    """Extract CUDA kernel intervals (start, end, stream_key) from trace events.

    - Only ``ph == "X"`` and ``cat == "kernel"`` entries are considered.
    - ``cudaLaunchKernel`` CPU-side runtime events are ignored by construction
      because they live in the ``cuda_runtime`` category.
    """
    intervals: List[Tuple[float, float, Tuple[Optional[int], Optional[int]]]] = []

    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue

        try:
            ts = float(ev.get("ts", 0.0))
            dur = float(ev.get("dur", 0.0))
        except (TypeError, ValueError):
            continue

        if dur <= 0.0:
            continue

        start = ts
        end = ts + dur

        args = ev.get("args", {}) or {}
        device = args.get("device")
        stream = args.get("stream", ev.get("tid"))
        stream_key = (device, stream)

        intervals.append((start, end, stream_key))

    intervals.sort(key=lambda x: x[0])
    return intervals


def _collect_memcpy_intervals(events: Iterable[Dict]) -> List[Tuple[float, float]]:
    """Extract CUDA memcpy intervals (start, end) from trace events."""
    intervals: List[Tuple[float, float]] = []

    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "gpu_memcpy":
            continue

        try:
            ts = float(ev.get("ts", 0.0))
            dur = float(ev.get("dur", 0.0))
        except (TypeError, ValueError):
            continue

        if dur <= 0.0:
            continue

        start = ts
        end = ts + dur
        intervals.append((start, end))

    intervals.sort(key=lambda x: x[0])
    return intervals


def _compute_total_busy_time_us(
    intervals: List[Tuple[float, float, Tuple[Optional[int], Optional[int]]]]
) -> float:
    """Compute wall-clock time where at least one kernel is running."""
    if not intervals:
        return 0.0

    total = 0.0
    cur_start, cur_end, _ = intervals[0]

    for start, end, _ in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            if end > cur_end:
                cur_end = end

    total += cur_end - cur_start
    return total


def _compute_concurrent_time_us(
    intervals: List[Tuple[float, float, Tuple[Optional[int], Optional[int]]]]
) -> float:
    """Compute time with kernels running concurrently on different streams."""
    if not intervals:
        return 0.0

    events: List[Tuple[float, int, Tuple[Optional[int], Optional[int]]]] = []
    for start, end, stream_key in intervals:
        events.append((start, +1, stream_key))
        events.append((end, -1, stream_key))

    # Sort by time; for ties, process end events (-1) before start events (+1)
    # so that back-to-back kernels do not count as overlapping.
    events.sort(key=lambda x: (x[0], x[1]))

    active_counts: Dict[Tuple[Optional[int], Optional[int]], int] = defaultdict(int)
    active_streams = 0
    prev_t: Optional[float] = None
    concurrent_time = 0.0

    for t, delta, stream_key in events:
        if prev_t is not None and active_streams >= 2:
            concurrent_time += t - prev_t

        if delta > 0:
            if active_counts[stream_key] == 0:
                active_streams += 1
            active_counts[stream_key] += 1
        else:
            if active_counts[stream_key] > 0:
                active_counts[stream_key] -= 1
                if active_counts[stream_key] == 0:
                    active_streams -= 1

        prev_t = t

    return concurrent_time


def _compute_kernel_with_memcpy_time_us(
    kernel_intervals: List[Tuple[float, float, Tuple[Optional[int], Optional[int]]]],
    memcpy_intervals: List[Tuple[float, float]],
) -> float:
    """Compute time with at least one kernel and one memcpy running."""
    if not kernel_intervals or not memcpy_intervals:
        return 0.0

    events: List[Tuple[float, int, str]] = []

    for start, end, _ in kernel_intervals:
        events.append((start, +1, "k"))
        events.append((end, -1, "k"))

    for start, end in memcpy_intervals:
        events.append((start, +1, "m"))
        events.append((end, -1, "m"))

    events.sort(key=lambda x: (x[0], x[1]))

    k_active = 0
    m_active = 0
    prev_t: Optional[float] = None
    overlap = 0.0

    for t, delta, kind in events:
        if prev_t is not None and k_active > 0 and m_active > 0:
            overlap += t - prev_t

        if kind == "k":
            k_active += delta
        else:
            m_active += delta

        prev_t = t

    return overlap


def compute_cuda_kernel_times(trace_path: TracePath) -> Tuple[float, float, float]:
    """Compute CUDA kernel execution, concurrency, and IO-overlap times.

    The input should be a Chrome/Perfetto JSON trace produced by torch.profiler,
    e.g. ``profile/trace.json``.

    Returns:
        (total_kernel, concurrent_kernel, kernel_with_memcpy)

        All values are in the same time units as the trace (typically
        microseconds):
        - ``total_kernel``: wall-clock time with at least one CUDA kernel
          running on any stream/device.
        - ``concurrent_kernel``: time with kernels running concurrently on
          at least two distinct streams.
        - ``kernel_with_memcpy``: time with at least one CUDA kernel and at
          least one CUDA memcpy (``gpu_memcpy``) active concurrently.
    """
    events = _load_trace_events(trace_path)
    kernel_intervals = _collect_kernel_intervals(events)
    memcpy_intervals = _collect_memcpy_intervals(events)
    total_busy = _compute_total_busy_time_us(kernel_intervals)
    concurrent = _compute_concurrent_time_us(kernel_intervals)
    kernel_with_memcpy = _compute_kernel_with_memcpy_time_us(kernel_intervals, memcpy_intervals)
    return total_busy, concurrent, kernel_with_memcpy


# Null context helper for call sites
def nullcontext():
    return _NullContext()
