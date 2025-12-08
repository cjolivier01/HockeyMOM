import cv2
import torch


def load_first_video_frame(filename: str) -> torch.Tensor:
    """Load the first frame of a video into a :class:`torch.Tensor`.

    @param filename: Path to the input video file.
    @return: Tensor of shape ``[H, W, C]`` in BGR order, or ``None`` if loading fails.
    @see @ref hmlib.utils.time.format_duration_to_hhmmss "format_duration_to_hhmmss"
         for formatting timestamps when inspecting videos.
    """
    cap = cv2.VideoCapture(filename)
    try:
        ok, img = cap.read()
        if ok:
            return torch.from_numpy(img)
    finally:
        if cap is not None:
            cap.release()
    return None
