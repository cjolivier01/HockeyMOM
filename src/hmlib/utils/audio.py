from typing import Optional, Tuple, Union

import ffmpegio
import numpy as np
import torch

def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor], duration_seconds: float, verbose: Optional[bool] = False
) -> Tuple[torch.Tensor, float]:
    """
    Loads audio from a file and returns it as a PyTorch tensor.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        waveform (torch.Tensor): The audio as a tensor [channels, samples].
        sample_rate (int): The sample rate of the audio.
    """
    smaples_per_second, waveform = ffmpegio.audio.read(audio, t=duration_seconds)
    if verbose:
        # The waveform is now a PyTorch tensor with shape [channels, samples]
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {smaples_per_second}")

    return waveform, smaples_per_second
