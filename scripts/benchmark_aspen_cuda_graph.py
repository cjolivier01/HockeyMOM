from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from hmlib.aspen import AspenNet
from hmlib.video.video_out import VideoOutput


@dataclass
class TimingStats:
    first_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkResult:
    benchmark: str
    mode: str
    iterations: int
    warmup: int
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    timing_ms: TimingStats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Aspen CUDA-graph vs non-CUDA-graph execution."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--input-width", type=int, default=4096)
    parser.add_argument("--input-height", type=int, default=2160)
    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=2160)
    parser.add_argument(
        "--run-command",
        type=str,
        default=None,
        help="Optional shell-style command to benchmark with and without --aspen-cuda-graph.",
    )
    parser.add_argument(
        "--run-repeats",
        type=int,
        default=1,
        help="Number of sequential runs for --run-command benchmarks.",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip the isolated synthetic CUDA benchmarks and only run --run-command.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON in addition to the table.")
    return parser.parse_args()


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        raise ValueError("samples must be non-empty")
    if len(samples) == 1:
        return samples[0]
    idx = (len(samples) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(samples) - 1)
    frac = idx - lo
    return float(samples[lo] * (1.0 - frac) + samples[hi] * frac)


def _summarize(first_ms: float, samples_ms: list[float]) -> TimingStats:
    ordered = sorted(float(x) for x in samples_ms)
    return TimingStats(
        first_ms=float(first_ms),
        mean_ms=float(statistics.mean(ordered)),
        median_ms=float(statistics.median(ordered)),
        p95_ms=float(_percentile(ordered, 0.95)),
        min_ms=float(min(ordered)),
        max_ms=float(max(ordered)),
    )


def _time_loop(
    fn: Callable[[int], Any],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> TimingStats:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn(0)
    torch.cuda.synchronize(device)
    first_ms = (time.perf_counter() - start) * 1000.0

    for step in range(1, 1 + warmup):
        fn(step)
    torch.cuda.synchronize(device)

    samples_ms: list[float] = []
    for step in range(1 + warmup, 1 + warmup + iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn(step)
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return _summarize(first_ms=first_ms, samples_ms=samples_ms)


def _make_input_tensor(height: int, width: int, device: torch.device) -> torch.Tensor:
    total = int(height) * int(width) * 3
    return torch.arange(total, device=device, dtype=torch.uint8).reshape(1, height, width, 3)


def _summarize_repeated_runs(samples_ms: list[float]) -> TimingStats:
    if not samples_ms:
        raise ValueError("samples_ms must be non-empty")
    ordered = sorted(float(x) for x in samples_ms)
    return _summarize(first_ms=samples_ms[0], samples_ms=ordered)


def _repo_pythonpath() -> str:
    repo_root = str(Path(__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return repo_root
    paths = existing.split(os.pathsep)
    if repo_root in paths:
        return existing
    return os.pathsep.join([repo_root, existing])


def _command_with_cuda_graph(base_command: str, *, enabled: bool) -> list[str]:
    argv = shlex.split(base_command)
    graph_flag = "--aspen-cuda-graph"
    has_flag = any(arg == graph_flag for arg in argv)
    if enabled and not has_flag:
        argv.append(graph_flag)
    if not enabled and has_flag:
        argv = [arg for arg in argv if arg != graph_flag]
    return argv


def _benchmark_run_command(
    *,
    base_command: str,
    cuda_graph: bool,
    repeats: int,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> BenchmarkResult:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    argv = _command_with_cuda_graph(base_command, enabled=cuda_graph)
    env = os.environ.copy()
    env["PYTHONPATH"] = _repo_pythonpath()
    samples_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="aspen-bench-run-") as tmp_dir:
        for repeat_idx in range(repeats):
            log_path = Path(tmp_dir) / f"{'graph' if cuda_graph else 'normal'}_{repeat_idx}.log"
            start = time.perf_counter()
            with open(log_path, "w", encoding="utf-8") as log_fp:
                proc = subprocess.run(
                    argv,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    env=env,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples_ms.append(elapsed_ms)
            if proc.returncode != 0:
                tail = log_path.read_text(encoding="utf-8")[-4000:]
                raise RuntimeError(
                    f"Benchmark command failed with exit code {proc.returncode}: "
                    f"{' '.join(argv)}\nLog tail:\n{tail}"
                )

    timing = _summarize_repeated_runs(samples_ms)
    return BenchmarkResult(
        benchmark="hm_run_command",
        mode="cuda_graph" if cuda_graph else "normal",
        iterations=repeats,
        warmup=0,
        input_width=input_width,
        input_height=input_height,
        output_width=output_width,
        output_height=output_height,
        timing_ms=timing,
    )


def _benchmark_video_output_prepare(
    *,
    cuda_graph: bool,
    warmup: int,
    iterations: int,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    device: torch.device,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="aspen-bench-video-") as tmp_dir:
        img = _make_input_tensor(height=input_height, width=input_width, device=device)
        video_out = VideoOutput(
            output_video_path=str(Path(tmp_dir) / "video_out.mkv"),
            fps=30.0,
            skip_final_save=True,
            device=device,
            output_width=output_width,
            output_height=output_height,
        )
        video_out.set_cuda_graph_enabled(cuda_graph)

        base_cfg = {
            "output_frame_width": input_width,
            "output_frame_height": input_height,
            "output_aspect_ratio": float(input_width) / float(input_height),
        }

        def _run(step: int) -> Any:
            results = {
                "img": img,
                "frame_ids": torch.tensor([step + 1], dtype=torch.int64),
                "video_frame_cfg": dict(base_cfg),
            }
            prepared = video_out.prepare_results(results)
            return prepared["img"]

        timing = _time_loop(fn=_run, warmup=warmup, iterations=iterations, device=device)
        return BenchmarkResult(
            benchmark="video_output_prepare",
            mode="cuda_graph" if cuda_graph else "normal",
            iterations=iterations,
            warmup=warmup,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            timing_ms=timing,
        )


def _benchmark_aspen_video_pipeline(
    *,
    cuda_graph: bool,
    warmup: int,
    iterations: int,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    device: torch.device,
) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="aspen-bench-pipeline-") as tmp_dir:
        img = _make_input_tensor(height=input_height, width=input_width, device=device)
        shared = {"device": device, "game_config": {}}
        graph_cfg = {
            "pipeline": {"cuda_graph": cuda_graph},
            "minimal_context": True,
            "plugins": {
                "video_out_prep": {
                    "class": "hmlib.aspen.plugins.video_out_plugin.VideoOutPrepPlugin",
                    "depends": [],
                    "params": {
                        "output_video_path": str(Path(tmp_dir) / "tracking.mkv"),
                        "skip_final_save": True,
                        "output_width": output_width,
                        "output_height": output_height,
                    },
                },
                "video_out": {
                    "class": "hmlib.aspen.plugins.video_out_plugin.VideoOutPlugin",
                    "depends": ["video_out_prep"],
                    "params": {
                        "output_video_path": str(Path(tmp_dir) / "tracking.mkv"),
                        "skip_final_save": True,
                        "output_width": output_width,
                        "output_height": output_height,
                    },
                },
            },
        }
        net = AspenNet("benchmark_video_pipeline", graph_cfg, shared=shared)
        base_cfg = {
            "output_frame_width": input_width,
            "output_frame_height": input_height,
            "output_aspect_ratio": float(input_width) / float(input_height),
        }

        def _run(step: int) -> Any:
            context = {
                "img": img,
                "frame_ids": torch.tensor([step + 1], dtype=torch.int64),
                "video_frame_cfg": dict(base_cfg),
                "work_dir": tmp_dir,
            }
            out = net(context)
            return out.get("img")

        try:
            timing = _time_loop(fn=_run, warmup=warmup, iterations=iterations, device=device)
        finally:
            net.finalize()

        return BenchmarkResult(
            benchmark="aspen_video_pipeline",
            mode="cuda_graph" if cuda_graph else "normal",
            iterations=iterations,
            warmup=warmup,
            input_width=input_width,
            input_height=input_height,
            output_width=output_width,
            output_height=output_height,
            timing_ms=timing,
        )


def _format_row(result: BenchmarkResult) -> str:
    t = result.timing_ms
    return (
        f"{result.benchmark:24} {result.mode:11} "
        f"{t.first_ms:10.3f} {t.mean_ms:10.3f} {t.median_ms:10.3f} "
        f"{t.p95_ms:10.3f} {t.min_ms:10.3f} {t.max_ms:10.3f}"
    )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda", torch.cuda.current_device())

    results: list[BenchmarkResult] = []
    if not args.skip_synthetic:
        results.extend(
            [
                _benchmark_video_output_prepare(
                    cuda_graph=False,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    device=device,
                ),
                _benchmark_video_output_prepare(
                    cuda_graph=True,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    device=device,
                ),
                _benchmark_aspen_video_pipeline(
                    cuda_graph=False,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    device=device,
                ),
                _benchmark_aspen_video_pipeline(
                    cuda_graph=True,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    device=device,
                ),
            ]
        )
    if args.run_command:
        results.extend(
            [
                _benchmark_run_command(
                    base_command=args.run_command,
                    cuda_graph=False,
                    repeats=args.run_repeats,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                ),
                _benchmark_run_command(
                    base_command=args.run_command,
                    cuda_graph=True,
                    repeats=args.run_repeats,
                    input_width=args.input_width,
                    input_height=args.input_height,
                    output_width=args.output_width,
                    output_height=args.output_height,
                ),
            ]
        )

    print(
        "benchmark                 mode          first_ms    mean_ms  median_ms     p95_ms     min_ms     max_ms"
    )
    for result in results:
        print(_format_row(result))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))


if __name__ == "__main__":
    main()
