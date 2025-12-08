import os

from torch import Tensor
from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder(
    f"{os.environ['HOME']}/Videos/dh-bh-1/tracking_output-with-audio.mp4", device="cuda:0"
)
print("Created decoder)")
# Access metadata (optional)
# metadata = decoder.metadata
# print(f"Video duration: {metadata.duration_seconds} seconds")
# print(f"Average FPS: {metadata.average_fps}")
# print(f"Video resolution: {metadata.width}x{metadata.height}")

# Index based frame retrieval.
# first_ten_frames: Tensor = decoder[10:]
# last_ten_frames: Tensor = decoder[-10:]

# Multi-frame retrieval, index and time based.
# frames = decoder.get_frames_at(indices=[10, 0, 15])
# frames = decoder.get_frames_played_at(seconds=[0.2, 3, 4.5])
