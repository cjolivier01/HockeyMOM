import cv2
import numpy as np
import torch
import torchaudio
import torchvision

from hmlib.tracking_utils.log import logger
from hmlib.utils.image import resize_image
from hmlib.tracking_utils.timer import Timer

from .ffmpeg import BasicVideoInfo, get_ffmpeg_decoder_process

#
# Live stream:
# https://www.digitalocean.com/community/tutorials/how-to-set-up-a-video-streaming-server-using-nginx-rtmp-on-ubuntu-22-04
#

_EXTENSION_MAPPING = {
    "matroska": "mkv",
}

_FOURCC_TO_CODEC = {
    "HEVC": "hevc_cuvid",
    "H264": "h264_cuvid",
    "MJPEG": "mjpeg_cuvid",
    "XVID": "mpeg4_cuvid",
    "MP4V": "mpeg4_cuvid",
    "FMP4": "mpeg4_cuvid",
}

MAX_VIDEO_WIDTH = 1280

def video_size(width: int, height: int, max_width: int = MAX_VIDEO_WIDTH):
    h = height
    w = width
    if h > max_width:
        ar = w / h
        if ar > 2.1:
            # probably the whole panorama
            max_width = max_width + 500
        w = int(max_width)
        h = int(w / ar)
        return w, h, True
    return w, h, False


def scale_down_for_live_video(tensor: torch.Tensor, max_width: int = MAX_VIDEO_WIDTH):
    assert tensor.ndim == 4 and (tensor.shape[-1] == 3 or tensor.shape[-1] == 4)
    h = tensor.shape[1]
    w = tensor.shape[2]
    w, h, resized = video_size(width=w, height=h, max_width=max_width)
    if resized:
        return resize_image(tensor, new_height=h, new_width=w)
    return tensor


class VideoStreamWriter:
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
        format: str = "bgr24",
        batch_size: int = 3,
        bit_rate: int = int(55e6),
        device: torch.device = None,
        lossless: bool = False,
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
                for i in np.linspace(
                    0, np.round(self._fps) - 1, self._stream_fps, endpoint=False
                )
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
            self._width, self._height, _ = video_size(
                width=self._width, height=self._height
            )

        self._video_out = None
        self._video_f = None
        self._device = device
        self._lossless = lossless
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
                if isinstance(image, torch.Tensor):
                    image = image.permute(0, 3, 1, 2)
                else:
                    image = image.transpose(0, 3, 1, 2)
        return image

    def _add_stream(self):
        if self._lossless:
            assert False
            preset = "lossless"
            rate_control = "constqp"
        else:
            # preset = "slow"
            preset = "p7"
            rate_control = "cbr"
        options = {
            # "preset": preset,
            "rc": rate_control,
            "minrate": "55M",
            "maxrate": "55M",
            "bufsize": "55M",
        }
        if self._lossless:
            options["qp"] = "0"

        if self._filename.startswith("rtmp://"):
            new_w, new_h, needs_resize = video_size(
                width=self._width, height=self._height
            )
            assert not self._local_resize or not needs_resize
            hw_accel = None
            if self._codec.endswith("_nvenc"):
                hw_accel = str(self._device)
            # Cut down the bitrate
            self._codec_config.bit_rate /= 4
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
                # encoder_frame_rate=self._fps,
                codec_config=self._codec_config,
                hw_accel=str(self._device),
            )
        print("Video stream added")

    def bgr_to_rgb(self, batch: torch.Tensor):
        # Assuming batch is a PyTorch tensor of shape [N, C, H, W]
        # and the channel order is BGR
        return batch[:, [2, 1, 0], :, :]

    def close(self):
        if self._video_f is not None:
            self._video_f.close()
            self._video_f = None

    def release(self):
        self.close()

    def flush(self, flush_video_file: bool = True):
        if self._batch_items:
            if len(self._batch_items[0].shape) == 3:
                image_batch = torch.stack(self._batch_items)
            else:
                image_batch = torch.cat(self._batch_items, dim=0)
            self._batch_items.clear()
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
        if len(self._batch_items) >= self._batch_size:
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

    def __next__(self):
        if self._batch_size == 1:
            res, frame = self._cap.read()
            if not res:
                raise StopIteration()
            return np.expand_dims(frame.transpose(2, 0, 1), axis=0)
        else:
            frames = []
            for _ in range(self._batch_size):
                res, frame = self._cap.read()
                if not res:
                    raise StopIteration()
                frames.append(frame.transpose(2, 0, 1))
            return np.stack(frames)


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
        assert len(next_chunk) == self._batch_size
        frame = next_chunk[0]
        return frame

    # def __next__(self):
    #     if self._chunk is None or self._chunk_position >= len(self._chunk):
    #         next_chunk = next(self._iter)
    #         if next_chunk is None:
    #             raise StopIteration()
    #         assert len(next_chunk) == 1
    #         self._chunk = next_chunk[0]
    #         self._chunk_position = 0
    #     frame = self._chunk[self._chunk_position]
    #     self._chunk_position += 1
    #     return frame.unsqueeze(0)


# def get_ffmpeg_decoder_process(
#     input_video: str,
#     gpu_index: int,
#     buffer_size=10**8,
#     loglevel: str = "quiet",
#     format: str = "bgr24",
#     time_s: float = 0.0,
# ):


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
            self._batch_size
            * self._vid_info.width
            * self._vid_info.height
            * self._channels
        )

        if not raw_image:
            self._process.terminate()
            raise StopIteration()

        # Transform the byte read into a numpy array
        # frame = np.frombuffer(raw_image, np.uint8).reshape(
        #     (self._vid_info.height, self._vid_info.width, self._channels)
        # )
        # frame = torch.from_numpy(frame)
        frame = (
            torch.frombuffer(buffer=raw_image, dtype=torch.uint8)
            # .to("cuda:0", non_blocking=False)
            .reshape(
                (
                    self._batch_size,
                    self._vid_info.height,
                    self._vid_info.width,
                    self._channels,
                )
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

    def __init__(
        self,
        filename: str,
        type: str,
        codec: str = None,
        batch_size: int = 1,
        device: torch.device = None,
    ):
        # subprocess_decode_ffmpeg(filename)
        self._filename = filename
        self._type = type
        self._codec = codec
        self._fps = None
        self._width = None
        self._height = None
        self._batch_size = batch_size
        if device is None:
            device = torch.device("cpu")
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self._video_in = None
        self._video_info = None
        self._iter = None
        self._ss = 0.0
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
        return self._video_info.bitrate

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
        self._video_in.add_basic_video_stream(
            frames_per_chunk=self._batch_size,
            stream_index=0,
            decoder_option={},
            format=self._device.type if self._device is not None else "cuda",
            hw_accel="cuda",
        )

    def isOpened(self):
        return self._video_in is not None

    def seek(self, timestamp: float = None, frame_number: int = None):
        assert timestamp is None or frame_number is None
        if frame_number is not None:
            timestamp = float(frame_number) / self.fps
        else:
            frame_number = timestamp * self.fps
        if isinstance(self._video_in, torchaudio.io.StreamReader):
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
        self._video_info = BasicVideoInfo(video_file=self._filename)
        if self._codec is None:
            self._codec = _FOURCC_TO_CODEC.get(self._video_info.codec, None)
            if self._codec is None and self._type != "cv2":
                print(
                    f"VideoStreamReader is changing decoder from {self._type} "
                    f"to cv2 due to video's codec type: {self._video_info.codec}"
                )
                self._type = "cv2"
        if self._type == "torchaudio":
            self._video_in = torchaudio.io.StreamReader(src=self._filename)
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
            if isinstance(self._video_in, torchaudio.io.StreamReader):
                self._video_in.remove_stream(0)
            elif isinstance(self._video_in, torchvision.io.VideoReader):
                pass
            elif isinstance(self._video_in, cv2.VideoCapture):
                self._video_in.release()
            elif isinstance(self._video_in, FFMpegVideoReader):
                pass
            else:
                assert False
            self._video_in = None
            self._iter = None
        return

    def read(self):
        next_data = next(self._iter)
        if next_data is None:
            return False, None
        return True, next_data
