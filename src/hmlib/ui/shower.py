import cv2
import numpy as np
import threading
from typing import Optional, Union
import torch

from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import (
    # CachedIterator,
    StreamCheckpoint,
    StreamTensor,
    # cuda_stream_scope,
    # get_gpu_capabilities,
)

from hmlib.utils.image import (
    # ImageColorScaler,
    # ImageHorizontalGaussianDistribution,
    # crop_image,
    # image_height,
    # image_width,
    # make_channels_last,
    make_visible_image,
    # resize_image,
)


class Shower:
    def __init__(self, show_scaled: Optional[float] = None):
        self._show_scaled = show_scaled
        self._q = create_queue(mp=False)
        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

    def _do_show(self, img: Union[torch.Tensor, np.ndarray]):
        if img.ndim == 3:
            if isinstance(img, torch.Tensor | StreamTensor):
                img = img.unsqueeze(0)
            elif isinstance(img, np.ndarray):
                img = np.expand_dims(img, axis=0)
        if isinstance(img, StreamTensor):
            img = img.get()
        for s_img in img:
            cv2.imshow(
                "online_im",
                make_visible_image(s_img, enable_resizing=self._show_scaled),
            )
            cv2.waitKey(1)

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None

    def _worker(self):
        while True:
            img = self._q.get()
            if img is None:
                break
            self._do_show(img=img)

    def show(self, img: Union[torch.Tensor, np.ndarray]):
        if self._thread is not None:
            self._q.put(img)
