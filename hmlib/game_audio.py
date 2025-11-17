"""Helpers for transferring audio into rendered game videos.

This module chooses a suitable output filename (usually under a game-specific
directory) and delegates to :func:`hmlib.audio.copy_audio` for the actual
audio copy or merge.

@see @ref hmlib.audio.copy_audio "copy_audio" for the lower-level ffmpeg wrapper.
@see @ref hmlib.config.get_game_dir "get_game_dir" for game directory resolution.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from hmlib.audio import copy_audio
from hmlib.config import get_game_dir
from hmlib.utils.path import add_suffix_to_filename


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
    @param output_av_path: Optional explicit output path; auto-generated when ``None``.
    @param max_iterations: Max attempts when searching for a free destination filename.
    @return: Path to the resulting video file with audio merged in.
    @see @ref hmlib.audio.copy_audio "copy_audio" for the underlying ffmpeg call.
    """
    if not output_av_path:
        game_video_dir = get_game_dir(game_id)
        # TODO: Use hmlib.utils.path functions for this (or create as needed)
        if not game_video_dir:
            # Going into results dir
            dir_tokens = video_source_file.split("/")
            file_name = dir_tokens[-1]
            fn_tokens = file_name.split(".")
            if len(fn_tokens) > 1:
                if fn_tokens[-1] == "mkv":
                    # There will be audio drift when adding audio
                    # from mkv to mkv due to strange frame rate items
                    # in the mkv that differ from the original
                    fn_tokens[-1] = "mp4"
                fn_tokens[-2] += "-with-audio"
            else:
                fn_tokens[0] += "-with-audio"
            dir_tokens[-1] = ".".join(fn_tokens)
            output_av_path = os.path.join(*dir_tokens)
        else:
            # Going into game-dir (numbered if pre-existing)
            file_name = video_source_file.split("/")[-1]
            file_name = add_suffix_to_filename(file_name, "-with-audio")
            base_name, extension = os.path.splitext(file_name)
            if extension == ".mkv":
                # There will be audio drift when adding audio
                # from mkv to mkv due to strange frame rate items
                # in the mkv that differ from the original
                extension = ".mp4"
            output_av_path = None
            for i in range(max_iterations):
                if i:
                    fname = base_name + "-" + str(i) + extension
                else:
                    fname = base_name + extension
                fname = os.path.join(game_video_dir, fname)
                if not os.path.exists(fname):
                    output_av_path = fname
                    break
            if output_av_path is None:
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
