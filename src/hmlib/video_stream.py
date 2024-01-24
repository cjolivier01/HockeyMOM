import torch
import torchaudio


_EXTENSION_MAPPING = {
    "matroska": "mkv",
}


class VideoStreamWriter:
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        codec: str,
        format: str = "bgr24",
        batch_size: int = 10,
        bit_rate: int = 5000000,
        device: torch.device = None,
        lossless: bool = True,
        container_type: str = "matroska",
    ):
        self._filename = filename
        self._container_type = container_type
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec
        self._format = format
        self._video_out = None
        self._video_f = None
        self._device = device
        self._lossless = lossless
        assert batch_size >= 1
        self._batch_size = batch_size
        self._batch_items = []
        self._in_flush = False
        self._codec_config = torchaudio.io.CodecConfig(
            bit_rate=bit_rate,
        )
        self._codec_config.bit_rate = bit_rate
        self._codec_config = None
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
                image = image.permute(2, 0, 1)
        else:
            if image.shape[-1] == 3 and self._device is not None:
                image = image.permute(0, 3, 1, 2)
        return image

    def _add_stream(self):
        if self._lossless:
            preset = "lossless"
            rate_control = "constqp"
        else:
            preset = "slow"
            rate_control = "cbr"
        options = {
            "preset": preset,
            "rc": rate_control,
        }
        if self._lossless:
            options["qp"] = "0"

        self._video_out.add_video_stream(
            frame_rate=self._fps,
            height=self._height,
            width=self._width,
            format=self._format,
            encoder=self._codec,
            encoder_format="bgr0",
            encoder_option=options,
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
        self._batch_items.append(self._make_proper_permute(images))
        if len(self._batch_items) >= self._batch_size:
            self.flush(flush_video_file=False)

    def write(self, images: torch.Tensor):
        return self.append(images)
