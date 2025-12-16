"""Small audio helpers used by analytics and stitching pipelines."""

from typing import Optional, Tuple, Union

import ffmpegio
import numpy as np
import torch

from hmlib.log import get_logger


def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor],
    duration_seconds: float,
    verbose: Optional[bool] = False,
) -> Tuple[torch.Tensor, float]:
    """Load audio from a file (or tensor) and return a waveform tensor.

    @param audio: File path, NumPy array or :class:`torch.Tensor` containing audio data.
    @param duration_seconds: Duration (in seconds) to read from the source.
    @param verbose: If ``True``, print basic debug information.
    @return: Tuple ``(waveform, sample_rate)`` with shape ``[channels, samples]``.
    @see @ref hmlib.audio.copy_audio "hmlib.audio.copy_audio" for higher-level audio handling.
    """
    smaples_per_second, waveform = ffmpegio.audio.read(audio, t=duration_seconds, show_log=True)
    if verbose:
        # The waveform is now a PyTorch tensor with shape [channels, samples]
        logger = get_logger(__name__)
        logger.info("Waveform shape: %s", waveform.shape)
        logger.info("Sample rate: %s", smaples_per_second)

    return waveform, smaples_per_second
