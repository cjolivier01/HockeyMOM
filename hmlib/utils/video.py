import cv2
import torch


def load_first_video_frame(filename: str) -> torch.Tensor:
    cap = cv2.VideoCapture(filename)
    try:
        ok, img = cap.read()
        if ok:
            return torch.from_numpy(img)
    finally:
        if cap is not None:
            cap.release()
    return None


