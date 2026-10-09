"""Helpers for transferring audio into rendered game videos.

This module chooses a suitable output filename (usually under a game-specific
directory) and delegates to :func:`hmlib.audio.copy_audio` for the actual
audio copy or merge.

@see @ref hmlib.audio.copy_audio "copy_audio" for the lower-level ffmpeg wrapper.
@see @ref hmlib.config.get_game_dir "get_game_dir" for game directory resolution.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union

from hmlib.audio import copy_audio
from hmlib.config import get_game_dir


def transfer_audio(
    game_id: str,
    input_av_files: Union[str, List[str]],
    video_source_file: str,
    output_av_path: Optional[str] = None,
    max_iterations: int = 1000,
) -> Path:
    """Attach audio from one or more source files to a rendered game video.

    @param game_id: Game identifier used to resolve the target directory.
    @param input_av_files: Single path or list of paths containing the audio to copy.
    @param video_source_file: Video file that should receive the audio track.
    @param output_av_path: Optional output basename or positive version; auto-generated when ``None``.
    @param max_iterations: Max attempts when searching for a free destination filename.
    @return: Path to the resulting video file with audio merged in.
    @see @ref hmlib.audio.copy_audio "copy_audio" for the underlying ffmpeg call.
    """
    if output_av_path:
        requested = Path(output_av_path)
        match = re.search(r"-(\d+)$", requested.stem)
        if match and int(match[1]) < 1:
            raise ValueError("Output audio version must start at one")
        if match and os.path.lexists(requested):
            raise FileExistsError(f"Output audio version already exists: {requested}")
    else:
        game_video_dir = get_game_dir(game_id)
        video = Path(video_source_file)
        directory = Path(game_video_dir) if game_video_dir else video.parent
        # Remux MKV to MP4 to avoid the audio drift of MKV-to-MKV copies.
        extension = ".mp4" if video.suffix == ".mkv" else video.suffix
        requested = directory / (video.stem + "-with-audio" + extension)
        match = None
    if match is None:
        # Work videos have stable names; published audio versions start at one.
        directory = requested.parent
        base_name, extension = requested.stem, requested.suffix
        pattern = re.compile(rf"^{re.escape(base_name)}-(\d+){re.escape(extension)}$")
        first_version = 1
        for existing in directory.iterdir():
            match = pattern.fullmatch(existing.name)
            if match:
                first_version = max(first_version, int(match[1]) + 1)
        for version in range(first_version, first_version + max_iterations):
            candidate = directory / f"{base_name}-{version}{extension}"
            if not os.path.lexists(candidate):
                output_av_path = str(candidate)
                break
        else:
            raise RuntimeError(
                f"Could not find a free destination file name after {max_iterations} iteration attempts"
            )
    print(f"Saving video with audio to file: {output_av_path}")

    copy_audio(
        input_audio=input_av_files,
        input_video=video_source_file,
        output_video=output_av_path,
    )
    return Path(output_av_path)
