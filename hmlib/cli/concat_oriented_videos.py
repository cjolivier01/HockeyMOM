#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

from hmlib.config import get_game_config_private, get_game_dir, get_nested_value


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


def _concat_copy(inputs: List[str], output_path: str) -> None:
    _require_ffmpeg()
    concat_list_path = _write_concat_file(inputs)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
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
        subprocess.run(cmd, check=True)
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
