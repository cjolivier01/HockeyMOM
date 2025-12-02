from torchcodec.decoders import VideoDecoder
from torch import Tensor

decoder = VideoDecoder("/mnt/home/colivier-local/Videos/dh-bh-1/tracking_output-with-audio.mp4")

# Index based frame retrieval.
first_ten_frames: Tensor = decoder[10:]
last_ten_frames: Tensor = decoder[-10:]

# Multi-frame retrieval, index and time based.
frames = decoder.get_frames_at(indices=[10, 0, 15])
frames = decoder.get_frames_played_at(seconds=[0.2, 3, 4.5])
