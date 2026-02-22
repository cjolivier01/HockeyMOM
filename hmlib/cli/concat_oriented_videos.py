#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.progress import BarColumn
    from rich.progress import Progress as RichProgress
    from rich.progress import SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - optional dependency during import
    Console = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    RichProgress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]

from hmlib.config import get_game_config_private, get_game_dir, get_nested_value

_RICH_CONSOLE = Console(stderr=True) if Console is not None else None
_FFMPEG_TIME_RE = re.compile(r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)$")


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def _resolve_inputs(game_id: str, game_dir: str, side: str) -> List[str]:
    cfg = get_game_config_private(game_id=game_id)
    rel_list = get_nested_value(cfg, f"game.videos.{side}")
    if not isinstance(rel_list, list) or not rel_list:
        raise RuntimeError(
            f"Missing game.videos.{side} in {os.path.join(game_dir, 'config.yaml')} "
            f"(run hmorientation --game-id {game_id})"
        )

    inputs: List[str] = []
    for item in rel_list:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        path = s if os.path.isabs(s) else os.path.join(game_dir, s)
        inputs.append(path)

    if not inputs:
        raise RuntimeError(
            f"Empty game.videos.{side} list in {os.path.join(game_dir, 'config.yaml')}"
        )

    missing = [p for p in inputs if not os.path.exists(p)]
    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(f"Missing input files for {side}:\n{missing_str}")

    return inputs


def _write_concat_file(paths: List[str]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for p in paths:
            abs_path = os.path.abspath(p)
            escaped = abs_path.replace("\\", "\\\\").replace("'", "\\'")
            tmp.write(f"file '{escaped}'\n")
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def _parse_ffmpeg_time_to_seconds(value: str) -> Optional[float]:
    match = _FFMPEG_TIME_RE.match(str(value).strip())
    if not match:
        return None
    try:
        hours = int(match.group("h"))
        minutes = int(match.group("m"))
        seconds = float(match.group("s"))
    except Exception:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _extract_progress_seconds(fields: Dict[str, str]) -> Optional[float]:
    time_value = fields.get("out_time")
    if time_value:
        parsed = _parse_ffmpeg_time_to_seconds(time_value)
        if parsed is not None:
            return parsed

    for key in ("out_time_us", "out_time_ms"):
        raw_value = fields.get(key)
        if raw_value is None:
            continue
        try:
            return float(int(raw_value)) / 1e6
        except Exception:
            continue

    return None


def _probe_duration_seconds(path: str) -> Optional[float]:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {path}: {details}")
    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"ffprobe returned empty duration for {path}")
    try:
        duration = float(text)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned invalid duration for {path}: {text!r}") from exc
    if duration <= 0:
        return None
    return duration


def _resolve_total_duration_seconds(inputs: List[str]) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None
    total = 0.0
    for path in inputs:
        try:
            duration = _probe_duration_seconds(path)
        except Exception as exc:
            print(
                f"Warning: unable to estimate total duration with ffprobe: {exc}",
                file=sys.stderr,
            )
            return None
        if duration is not None:
            total += duration
    return total if total > 0 else None


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def _run_ffmpeg_concat_with_progress(
    cmd: List[str],
    output_path: str,
    total_seconds: Optional[float],
) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    progress_fields: Dict[str, str] = {}
    ffmpeg_output: List[str] = []

    use_rich = (
        RichProgress is not None
        and TextColumn is not None
        and TimeElapsedColumn is not None
        and _RICH_CONSOLE is not None
    )
    if total_seconds and use_rich and BarColumn is not None and TimeRemainingColumn is not None:
        progress = RichProgress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[out_time]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_RICH_CONSOLE,
            transient=True,
        )
        task_id = progress.add_task(
            f"Concatenating {os.path.basename(output_path)}",
            total=total_seconds,
            out_time="00:00:00.00",
        )
    elif use_rich and SpinnerColumn is not None:
        progress = RichProgress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TextColumn("{task.fields[out_time]}"),
            TimeElapsedColumn(),
            console=_RICH_CONSOLE,
            transient=True,
        )
        task_id = progress.add_task(
            f"Concatenating {os.path.basename(output_path)}",
            total=None,
            out_time="00:00:00.00",
        )
    else:
        progress = None
        task_id = None

    if progress is None:
        print(f"Concatenating {os.path.basename(output_path)}...", file=sys.stderr)
    else:
        progress.start()

    try:
        stream = proc.stdout
        if stream is not None:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                if line.count("=") == 1 and " " not in line.split("=", 1)[0]:
                    key, value = line.split("=", 1)
                    progress_fields[key.strip()] = value.strip()
                    if key.strip() == "progress":
                        seconds = _extract_progress_seconds(progress_fields)
                        if seconds is not None:
                            out_time = _format_seconds(seconds)
                            if progress is not None and task_id is not None:
                                update_kwargs = {"out_time": out_time}
                                if total_seconds:
                                    update_kwargs["completed"] = min(seconds, total_seconds)
                                progress.update(task_id, **update_kwargs)
                            else:
                                print(f"  progress {out_time}", file=sys.stderr)
                        progress_fields = {}
                    continue
                ffmpeg_output.append(line)

        return_code = proc.wait()
        if return_code != 0:
            details = "\n".join(ffmpeg_output[-20:])
            raise subprocess.CalledProcessError(return_code, cmd, output=details)

        if progress is not None and task_id is not None and total_seconds:
            progress.update(
                task_id, completed=total_seconds, out_time=_format_seconds(total_seconds)
            )
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if progress is not None:
            progress.stop()


def _concat_copy(inputs: List[str], output_path: str) -> None:
    _require_ffmpeg()
    concat_list_path = _write_concat_file(inputs)
    try:
        total_seconds = _resolve_total_duration_seconds(inputs)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-nostats",
            "-progress",
            "pipe:1",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            output_path,
        ]
        _run_ffmpeg_concat_with_progress(
            cmd=cmd,
            output_path=output_path,
            total_seconds=total_seconds,
        )
    except Exception:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        raise
    finally:
        try:
            os.remove(concat_list_path)
        except Exception:
            pass


def concat_oriented_videos(game_id: str, do_left: bool, do_right: bool) -> None:
    game_dir = get_game_dir(game_id=game_id)
    if not game_dir:
        raise RuntimeError(f"Could not resolve game dir for game_id={game_id!r}")

    todo: List[str] = []
    if do_left:
        todo.append("left")
    if do_right:
        todo.append("right")
    if not todo:
        return

    for side in todo:
        output_path = os.path.join(game_dir, f"{side}.mp4")
        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
        inputs = _resolve_inputs(game_id=game_id, game_dir=game_dir, side=side)
        print(f"Concatenating {len(inputs)} files -> {output_path}")
        _concat_copy(inputs=inputs, output_path=output_path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate oriented left/right chapter lists from the game's private config "
            "into left.mp4/right.mp4 without re-encoding."
        )
    )
    parser.add_argument("--game-id", required=True, help="Game ID (directory under $HOME/Videos)")
    parser.add_argument("--left", action="store_true", help="Only build left.mp4")
    parser.add_argument("--right", action="store_true", help="Only build right.mp4")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    do_left = bool(args.left)
    do_right = bool(args.right)
    if not do_left and not do_right:
        do_left = True
        do_right = True
    try:
        concat_oriented_videos(game_id=str(args.game_id), do_left=do_left, do_right=do_right)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
