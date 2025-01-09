import os
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchaudio
import torchvision
from typeguard import typechecked

from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_channels_last, resize_image
from hmlib.video.ffmpeg import BasicVideoInfo, get_ffmpeg_decoder_process

#
# Live stream:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-a-video-streaming-server-using-nginx-rtmp-on-ubuntu-22-04
#

_EXTENSION_MAPPING = {
    "matroska": "mkv",
    "mp4": "mp4",
}

_FOURCC_TO_CODEC = {
    "HEVC": "hevc_cuvid",
    "HVC1": "hevc_cuvid",
    "HEV1": "hevc_cuvid",
    "H264": "h264_cuvid",
    "MJPEG": "mjpeg_cuvid",
    "XVID": "mpeg4_cuvid",
    "MP4V": "mpeg4_cuvid",
    "FMP4": "mpeg4_cuvid",
}

MAX_VIDEO_WIDTH = 8000  # 8K is 7680 x 4320
MAX_NEVC_VIDEO_WIDTH: int = 8192

# MAX_PIXEL_AREA: int = int(7680 * 4320)


def _max_video_width(codec: str) -> int:
    if "hevc" in codec:
        return MAX_NEVC_VIDEO_WIDTH
    return MAX_VIDEO_WIDTH


class VideoStreamWriterInterface:
    # TODO: Add the interface stubs
    pass


# def clamp_image_to_max_area(
#     width: int, height: int, max_area: int = MAX_PIXEL_AREA
# ) -> Tuple[int, int, bool]:
#     # Calculate the current area
#     current_area: int = width * height

#     # Check if the current area is within the limit
#     if current_area <= max_area:
#         return width, height, False  # No scaling needed

#     # Calculate the scaling factor
#     scaling_factor: float = (
#         max_area / current_area
#     ) ** 0.5  # Square root of the ratio of max_area to current_area

#     # Calculate new dimensions
#     new_width = int(width * scaling_factor)
#     new_height = int(height * scaling_factor)

#     return new_width, new_height, True


def video_size(
    width: int, height: int, codec: Optional[str] = None, max_width: Optional[int] = None
) -> Tuple[int | Literal[True]] | Tuple[int | Literal[False]]:
    assert codec or max_width
    if codec:
        assert not max_width
        max_width = _max_video_width(codec)
    h = height
    w = width
    if w > max_width:
        ar = w / h
        if ar > 2.1:
            # probably the whole panorama
            max_width = max_width + 500
        w = int(max_width)
        h = int(w / ar)
        return w, h, True
    return w, h, False


def clamp_max_video_dimensions(
    width: torch.Tensor,
    height: torch.Tensor,
    codec: Optional[str] = None,
    max_width: Union[None, int, torch.Tensor] = None,
) -> Tuple[int, int]:
    assert codec or max_width
    if codec:
        assert not max_width
        max_width = _max_video_width(codec)
    if width.ndim == 0:
        wh = torch.tensor([width, height])
    else:
        wh = torch.cat([width, height])
    wh_f = wh.to(torch.float)
    new_width = torch.ones_like(wh[0]) * max_width
    new_height = new_width.to(torch.float) / (wh_f[0] / wh_f[1])
    result_wh = torch.where(
        wh[0] <= new_width, wh, torch.tensor([new_width, new_height.to(new_width.dtype)])
    )
    return result_wh[0], result_wh[1]


def scale_down_for_live_video(tensor: torch.Tensor, max_width: int = MAX_VIDEO_WIDTH):
    assert tensor.ndim == 4 and (tensor.shape[-1] == 3 or tensor.shape[-1] == 4)
    h = tensor.shape[1]
    w = tensor.shape[2]
    w, h, resized = video_size(width=w, height=h, max_width=max_width)
    if resized:
        return resize_image(tensor, new_height=h, new_width=w)
    return tensor


def yuv_to_bgr_float(
    frames: torch.Tensor, dtype: torch.dtype = torch.float, non_blocking: bool = True
):
    """
    Current HW decode returns only YUV
    """
    if not torch.is_floating_point(frames):
        frames = frames.to(dtype, non_blocking=non_blocking)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([b, g, r], -1)
    rgb = (rgb * 255).clamp(0, 255)
    return rgb


def time_to_frame(time_str: str, fps: float):
    # Split the time duration string into components
    h = 0
    m = 0
    s = 0
    tokens = time_str.split(":")
    s = float(tokens[-1])
    if len(tokens) > 1:
        m = int(tokens[-2])
        if len(tokens) > 2:
            assert len(tokens) == 3
            h = int(tokens[0])
    # Extract seconds and milliseconds
    # Convert hours, minutes, seconds, and milliseconds to total seconds
    total_seconds = h * 3600 + m * 60 + s
    # Calculate the frame number
    frame_number = int(total_seconds * fps)
    return frame_number


class VideoStreamWriter(VideoStreamWriterInterface):

    @typechecked
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
        format: str = "bgr24",
        batch_size: int = 3,
        cache_size: int = 4,
        bit_rate: int = int(55e6),
        device: torch.device = None,
        lossless: bool = True,
        container_type: str = "matroska",
        local_resize: bool = True,
        streaming_drop_frame_interval: int = 3,
        stream_fps: int = 15,
    ):
        self._filename = filename
        self._container_type = container_type
        self._fps = fps

        self._stream_fps = stream_fps
        self._stream_frame_indexes = set(
            [
                int(i)
                for i in np.linspace(0, np.round(self._fps) - 1, self._stream_fps, endpoint=False)
            ]
        )
        self._width = width
        self._height = height
        self._streaming_drop_frame_interval = streaming_drop_frame_interval
        self._streaming_drop_frame_interval_counter = 0
        self._codec = codec
        self._streaming = False
        self._local_resize = local_resize
        self._timer = Timer()
        if self._filename.startswith("rtmp://"):
            self._format = format
            self._codec = "h264_nvenc"
            self._container_type = "flv"
            self._streaming = True
        elif self._filename.startswith("udp://"):
            self._format = "mpegts"
            self._container_type = "mpegts"
            self._streaming = True
        else:
            self._format = format

        if self._streaming and self._local_resize:
            self._width, self._height, _ = video_size(width=self._width, height=self._height)

        self._video_out = None
        self._video_f = None
        self._device = device
        self._lossless = lossless
        self._cache_size = cache_size
        assert batch_size >= 1
        self._batch_size = batch_size
        self._batch_count = 0
        self._batch_items = []
        self._in_flush = False
        self._codec_config = torchaudio.io.CodecConfig(
            bit_rate=bit_rate,
        )
        self._frame_counter = 0

    def __enter__(self):
        if self._video_f is None:
            self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _make_proper_permute(self, image: torch.Tensor):
        if len(image.shape) == 3:
            if image.shape[-1] == 3 and self._device is not None:
                if isinstance(image, torch.Tensor):
                    image = image.permute(2, 0, 1)
                else:
                    image = image.transpose(2, 0, 1)
        else:
            if image.shape[-1] == 3 and self._device is not None:
                if not isinstance(image, np.ndarray):
                    image = image.permute(0, 3, 1, 2)
                else:
                    image = image.transpose(0, 3, 1, 2)
        assert image.shape[-2] == self._height
        assert image.shape[-1] == self._width
        return image

    def _add_stream(self):
        if self._lossless:
            preset = "lossless"
            rate_control = "constqp"
        else:
            preset = "p7"
            rate_control = "cbr"
        options = {
            "rc": rate_control,
            "minrate": "55M",
            "maxrate": "55M",
            "bufsize": "55M",
            # "preset": preset,
        }
        if self._lossless:
            options["qp"] = "0"

        if self._filename.startswith("rtmp://"):
            new_w, new_h, needs_resize = video_size(width=self._width, height=self._height)
            assert not self._local_resize or not needs_resize
            hw_accel = None
            if self._codec.endswith("_nvenc"):
                hw_accel = str(self._device)
            # Cut down the bit_rate
            self._codec_config.bit_rate //= 4
            self._video_out.add_video_stream(
                frame_rate=self._stream_fps,
                format="bgr24",
                encoder=self._codec,
                encoder_format="bgr0",
                encoder_width=self._width,
                encoder_height=self._height,
                height=new_h,
                width=new_w,
                codec_config=self._codec_config,
                hw_accel=hw_accel,
            )
        else:
            self._video_out.add_video_stream(
                frame_rate=self._fps,
                height=self._height,
                width=self._width,
                format=self._format,
                encoder=self._codec,
                encoder_format="bgr0",
                encoder_option=options,
                encoder_frame_rate=self._fps,
                codec_config=self._codec_config,
                hw_accel=str(self._device),
            )

    def bgr_to_rgb(self, batch: torch.Tensor):
        # Assuming batch is a PyTorch tensor of shape [N, C, H, W]
        # and the channel order is BGR
        return batch[:, [2, 1, 0], :, :]

    def close(self):
        if self._video_f is not None:
            self._finish()
            self._video_f.close()
            self._video_f = None
            print(f"VideoStreamWriter wrote {self._frame_counter} frames")

    def release(self):
        self.close()

    def _finish(self):
        self.flush(flush_video_file=True, flush_all=True)

    def flush(self, flush_video_file: bool = True, flush_all: bool = False):
        def _get_tensor(t):
            if isinstance(t, StreamTensor):
                # return t.get()
                return t.wait()
            return t

        if self._batch_items:
            if flush_all:
                batch_items = self._batch_items
                self._batch_items = []
                if not batch_items:
                    return
            elif len(self._batch_items) <= self._cache_size:
                # Don't have at least cache_size items yet
                return
            else:
                flush_item_count = len(self._batch_items) - self._cache_size
                assert flush_item_count
                batch_items = self._batch_items[0:flush_item_count]
                self._batch_items = self._batch_items[flush_item_count:]
            if not batch_items:
                return
            batch_items = [_get_tensor(img) for img in batch_items]
            if len(batch_items) == 1:
                image_batch = batch_items[0]
                if len(image_batch.shape) == 3:
                    image_batch = image_batch.unsqueeze(0)
            else:
                if len(batch_items[0].shape) == 3:
                    image_batch = torch.stack(batch_items)
                else:
                    image_batch = torch.cat(batch_items, dim=0)
            frame_count = len(image_batch)
            self._video_out.write_video_chunk(
                i=0,
                chunk=image_batch,
            )
            self._frame_counter += frame_count

        if flush_video_file and self._video_f is not None:
            self._video_f.flush()

    def isOpened(self):
        return self._video_f is not None

    def open(self):
        assert self._video_f is None
        if not self._streaming:
            ext = _EXTENSION_MAPPING.get(self._container_type, self._container_type)
            if not self._filename.endswith("." + ext):
                self._filename += "." + ext
        self._video_out = torchaudio.io.StreamWriter(
            dst=self._filename, format=self._container_type
        )
        self._add_stream()
        self._video_f = self._video_out.open()

    def set(self, key: int, value: any):
        pass

    def get(self, key: int) -> any:
        return None

    def append(self, images: torch.Tensor):
        assert images.device == self._device
        if self._streaming:
            frame = self._streaming_drop_frame_interval_counter
            self._streaming_drop_frame_interval_counter += 1
            if (frame % int(self._fps)) not in self._stream_frame_indexes:
                assert images.shape[0] == 1
                return
            if self._local_resize:
                images = scale_down_for_live_video(images)
            assert images.device == self._device
            if not self._codec.endswith("_nvenc") and images.device.type != "cpu":
                images = images.cpu()
        self._batch_items.append(self._make_proper_permute(images))
        self.flush(flush_video_file=False)
        self._batch_count += 1
        if self._timer.start_time != 0:
            self._timer.toc()
        if self._batch_count % 100 == 0:
            logger.info(
                "Writing output ({:.2f} fps)".format(
                    len(images) * 1.0 / max(1e-5, self._timer.average_time),
                )
            )
            self._timer = Timer()
        self._timer.tic()

    def write(self, images: torch.Tensor):
        return self.append(images)


class CVVideoCaptureIterator:
    def __init__(self, cap: cv2.VideoCapture, batch_size: int = 1):
        self._cap = cap
        self._batch_size = batch_size
        self.frames_delivered_count = 0

    def __next__(self):
        if self._batch_size == 1:
            res, frame = self._cap.read()
            if not res:
                raise StopIteration()
            self.frames_delivered_count += 1
            return np.expand_dims(frame.transpose(2, 0, 1), axis=0)
        else:
            frames = []
            for _ in range(self._batch_size):
                res, frame = self._cap.read()
                if not res:
                    raise StopIteration()
                frames.append(frame.transpose(2, 0, 1))
            self.frames_delivered_count += self._batch_size
            return np.stack(frames)

    def __del__(self):
        if hasattr(self, "frames_delivered_count") and self.frames_delivered_count:
            logger.info(f"CVVideoCaptureIterator delivered {self.frames_delivered_count} frames")


class VideoReaderIterator:
    def __init__(self, vr: torchvision.io.VideoReader, batch_size: int = 1):
        self._vr = vr
        self._batch_size = batch_size

    def __next__(self):
        assert self._batch_size == 1
        next_frame = next(self._vr)
        if next_frame is None:
            raise StopIteration()
        return next_frame["data"].unsqueeze(0)


class TAStreamReaderIterator:
    def __init__(self, sr: torchaudio.io.StreamReader, batch_size: int = 1):
        self._iter = sr.stream()
        self._chunk = None
        self._chunk_position = 0
        self._batch_size = batch_size

    def __next__(self):
        next_chunk = next(self._iter)
        if next_chunk is None:
            raise StopIteration()
        frame = next_chunk[0]
        assert len(frame) == self._batch_size
        frame = yuv_to_bgr_float(frame)
        return frame


class FFMpegVideoReader:
    pass


class FFmpegVideoReaderIterator:
    def __init__(
        self,
        filename: str,
        gpu_index: int,
        time_s: float,
        format: str,
        vid_info: BasicVideoInfo,
        batch_size: int = 1,
    ):
        self._process = get_ffmpeg_decoder_process(
            input_video=filename, gpu_index=gpu_index, format=format, time_s=time_s
        )
        assert self._process is not None
        self._vid_info = vid_info
        self._channels = 3
        self._count = 0
        self._batch_size = batch_size

    def __del__(self):
        self._process.terminate()

    def __next__(self):
        if self._count:
            # Skip to the next frame in the buffer
            self._process.stdout.flush()
        raw_image = self._process.stdout.read(
            self._batch_size * self._vid_info.width * self._vid_info.height * self._channels
        )

        if not raw_image:
            self._process.terminate()
            raise StopIteration()

        frame = torch.frombuffer(buffer=raw_image, dtype=torch.uint8).reshape(
            (
                self._batch_size,
                self._vid_info.height,
                self._vid_info.width,
                self._channels,
            )
        )
        self._count += 1
        # Make channels-first
        return frame.permute(0, 3, 1, 2)


#
# VideoStreamReader
#
class VideoStreamReader:
    # type: str = "torchaudio",
    # type: str = "torchvision",
    # type: str = "cv2",
    # type: str = "ffmpeg",
    # type: str = "gstreamer",

    def __init__(
        self,
        filename: str,
        type: str,
        codec: str = None,
        batch_size: int = 1,
        device: torch.device = None,
    ):
        self._filename = filename
        self._type = type
        self._codec = codec
        self._fps = None
        self._width = None
        self._height = None
        self._batch_size = batch_size
        if device is None:
            device = torch.device("cpu")
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._video_in = None
        self._video_info = None
        self._iter = None
        self._ss = 0.0
        self._torchaudio_stream = False
        self._frames_delivered_count = 0
        self.open()

    @property
    def fps(self):
        return self._video_info.fps

    @property
    def width(self):
        return self._video_info.width

    @property
    def height(self):
        return self._video_info.height

    @property
    def bit_rate(self):
        return self._video_info.bit_rate

    @property
    def frame_count(self):
        return self._video_info.frame_count

    @property
    def codec(self):
        return self._video_info.codec

    def __len__(self):
        return self.frame_count

    def __iter__(self):
        if self._type == "torchaudio":
            return TAStreamReaderIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "torchvision":
            return VideoReaderIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "cv2":
            return CVVideoCaptureIterator(self._video_in, batch_size=self._batch_size)
        elif self._type == "ffmpeg":
            return FFmpegVideoReaderIterator(
                filename=self._filename,
                gpu_index=self._device.index if self._device is not None else 0,
                time_s=self._ss,
                format="bgr24",
                vid_info=self._video_info,
                batch_size=self._batch_size,
            )
        else:
            assert False

    def _add_stream(self):
        # Add the video stream
        hw_accel = None
        format = None
        decoder = None
        decoder_options: Dict[str, str] = {}
        if self._device.type == "cuda":
            # hw_accel = "cuda"
            decoder = _FOURCC_TO_CODEC[self._video_info.codec.upper()]
            hw_accel = str(self._device)
            # decoder_options["gpu"] = str(self._device.index)
            # format = "bgr24"
            # format = "rgb24"
            # format = "yuvj420p"
        self._video_in.add_basic_video_stream(
            frames_per_chunk=self._batch_size,
            stream_index=0,
            decoder=decoder,
            decoder_option=decoder_options,
            format=format,
            hw_accel=hw_accel,
        )

    def isOpened(self):
        return self._video_in is not None

    def seek(self, timestamp: float = None, frame_number: int = None):
        assert timestamp is None or frame_number is None
        if frame_number is not None:
            timestamp = float(frame_number) / self.fps
        else:
            frame_number = timestamp * self.fps
        if self._torchaudio_stream:
            self._video_in.seek(timestamp=timestamp, mode="precise")
        elif isinstance(self._video_in, torchvision.io.VideoReader):
            self._video_in.seek(time_s=timestamp)
        elif isinstance(self._video_in, cv2.VideoCapture):
            self._video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif isinstance(self._video_in, FFMpegVideoReader):
            self._ss = timestamp
        else:
            assert False

    def set(self, prop: int, value: any):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seek(frame_number=value)
        else:
            assert False and f"Unsupported property: {prop}"

    def get(self, prop: int):
        assert False
        return None

    def open(self):
        assert self._video_in is None
        self._video_info = BasicVideoInfo(self._filename)
        if self._codec is None and self._type != "ffmpeg":
            self._codec = _FOURCC_TO_CODEC.get(self._video_info.codec.upper(), None)
            if self._codec is None and self._type != "cv2":
                print(
                    f"VideoStreamReader is changing decoder from {self._type} "
                    f"to cv2 due to video's codec type: {self._video_info.codec}"
                )
                self._type = "cv2"
        if self._type == "torchaudio":
            self._video_in = torchaudio.io.StreamReader(src=self._filename)
            self._torchaudio_stream = True
            self._add_stream()
        elif self._type == "torchvision":
            self._video_in = torchvision.io.VideoReader(
                src=self._filename, stream="video", num_threads=32
            )
            self._meta = self._video_in.get_metadata()
        elif self._type == "cv2":
            self._video_in = cv2.VideoCapture(self._filename)
            if not self._video_in.isOpened():
                self._video_in.release()
                self._video_in = None
        elif self._type == "ffmpeg":
            self._video_in = FFMpegVideoReader()
        else:
            assert False

    def close(self):
        if self._video_in is not None:
            if self._torchaudio_stream:
                self._video_in.remove_stream(0)
            elif isinstance(self._video_in, torchvision.io.VideoReader):
                pass
            elif isinstance(self._video_in, cv2.VideoCapture):
                self._video_in.release()
            elif isinstance(self._video_in, FFMpegVideoReader):
                pass
            else:
                assert False
            if self._video_in is not None and hasattr(self._video_in, "frames_delivered_count"):
                print(f"VideoStreamReader delivered {self._video_in.frames_delivered_count} frames")
            self._video_in = None
            self._iter = None
        return

    def read(self):
        next_data = next(self._iter)
        if next_data is None:
            return False, None
        return True, next_data


class VideoStreamWriterCV2(VideoStreamWriterInterface):

    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        device: torch.device,
        codec: Optional[str],
        bit_rate: Optional[int] = None,
        batch_size: Optional[int] = 1,
    ):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._bit_rate = bit_rate
        self._output_video = cv2.VideoWriter(
            filename=filename,
            fourcc=fourcc,
            fps=fps,
            frameSize=(
                int(width),
                int(height),
            ),
        )
        self._output_video.set(
            cv2.CAP_PROP_BITRATE,
            self.calculate_desired_bitrate(width=width, height=height),
        )
        assert self._output_video.isOpened()
        self._device = torch.device("cpu")

    def isOpened(self) -> bool:
        return self._output_video is not None and self._output_video.isOpened()

    def calculate_desired_bitrate(self, width: int, height: int):
        # 4K @ 55M
        desired_bit_rate = self._bit_rate
        if not desired_bit_rate:
            desired_bit_rate_per_pixel = 55e6 / (3840 * 2160)
            desired_bit_rate = int(desired_bit_rate_per_pixel * width * height)
            logger.info(
                f"Desired bit rate for output video ({int(width)} x {int(height)}): {desired_bit_rate//1000} kb/s"
            )
        return desired_bit_rate

    def open(self):
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    def flush(self):
        pass

    def close(self):
        self._output_video.release()
        self._output_video = None

    def write(self, img: Union[torch.Tensor, np.ndarray]):
        if isinstance(img, StreamTensor):
            img = img.get()
        if img.ndim == 4:
            assert img.shape[0] == 1  # batch size of one only
            img = img.squeeze(0)
        # OpenCV always wants channels last
        img = make_channels_last(img)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            img = np.ascontiguousarray(img)
        if isinstance(img, np.ndarray) and img.dtype != np.dtype("uint8"):
            img = img.astype("uint8")
        self._output_video.write(img)


@typechecked
def create_output_video_stream(
    filename: str,
    fps: float,
    width: int,
    height: int,
    codec: Optional[str],
    device: torch.device,
    bit_rate: int = int(55e6),
    batch_size: Optional[int] = 1,
) -> VideoStreamWriterInterface:
    if "_nvenc" in codec or filename.startswith("rtmp://"):
        output_video = VideoStreamWriter(
            filename=filename,
            fps=fps,
            height=int(height),
            width=int(width),
            codec="hevc_nvenc",
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
        )
        output_video.open()
    else:
        output_video = VideoStreamWriterCV2(
            filename=filename,
            fps=fps,
            height=int(height),
            width=int(width),
            codec=codec,
            bit_rate=bit_rate,
            device=device,
            batch_size=batch_size,
        )
        output_video.open()
    return output_video


def extract_frame_image(
    source_video: str,
    frame_number: float,
    dest_image: str,
    stream_type: str = "cv2",
    # stream_type: str = "torchaudio",
) -> Union[torch.Tensor, np.ndarray]:
    print(f"Extracting frame {frame_number} from {source_video}...")
    reader = VideoStreamReader(
        filename=source_video,
        type=stream_type,
        device=torch.device("cuda", 0) if stream_type == "torchaudio" else None,
    )
    try:
        if frame_number:
            reader.seek(frame_number=frame_number)
        iterator = iter(reader)
        image = next(iterator)
    finally:
        reader.close()
    image = make_channels_last(image)
    if dest_image:
        img = image
        if isinstance(image, torch.Tensor):
            img = img.cpu().numpy()
        assert img.ndim == 4 and img.shape[0] == 1
        img = np.squeeze(img, axis=0)
        cv2.imwrite(dest_image, img)
    return image


if __name__ == "__main__":

    reader = VideoStreamReader(
        filename=f"{os.environ['HOME']}/Videos/ev-bs-short/GX010005.MP4",
        type="torchaudio",
        device=torch.device("cuda", 0),
    )
    iterator = iter(reader)
    image = next(iterator)
    show_image("image", image, wait=True)

    print("Done")
