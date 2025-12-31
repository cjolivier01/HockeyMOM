"""Asynchronous image viewer for debugging and live previews.

`Shower` consumes tensors or numpy arrays from a queue and displays them
via OpenCV or (optionally) Tkinter windows.

@see @ref hmlib.ui.show "show" for simpler one-off display helpers.
"""

import contextlib
import threading
import time
import tkinter as tk
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from hmlib.log import get_root_logger
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import StreamTensorBase, cuda_stream_scope, unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_last, make_visible_image
from hockeymom.core import show_cuda_tensor

from .show import cv2_has_opengl, show_gpu_tensor

# from .tk import get_tk_root


class ImageDisplayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        # Initially set up with an empty image
        self.image_label = tk.Label(master, text="No Image")
        self.image_label.pack()

    def display(self, tensor):
        # Convert the PyTorch tensor to a PIL Image
        image = Image.fromarray(make_channels_last(tensor).cpu().numpy().astype("uint8"))

        # Convert the PIL Image to a format Tkinter can use
        tk_image = ImageTk.PhotoImage(image)

        # Update the label with the new image
        self.image_label.configure(image=tk_image)

        # Keep a reference, avoid garbage collection
        self.image_label.image = tk_image


class Shower:

    def __init__(
        self,
        label: str,
        show_scaled: Optional[float] = None,
        max_size: int = 1,
        fps: Union[float, None] = None,
        cache_on_cpu: bool = False,
        logger=None,
        use_tk: bool = False,
        allow_gpu_gl: bool = True,
        profiler: Any = None,
        step: int = 1,
    ):
        self._label = label
        self._allow_gpu_gl = allow_gpu_gl
        self._show_scaled = show_scaled
        self._max_size: int = max(max_size, 1)
        self._fps = fps
        self._cache_on_cpu = cache_on_cpu
        self._stream = None
        self._profiler = profiler
        if self._fps is not None:
            self._label += " (" + str(self._fps) + " fps)"
        self._cv2_has_opengl_support = cv2_has_opengl()
        # TODO: use th
        self._step: int = step
        self._iter: int = 0
        self._next_frame_time = None
        self._use_tk = use_tk
        self._tk_displayer = None
        self._logger = logger if logger is not None else get_root_logger()
        self._q = create_queue(mp=False)
        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

    def _prof_ctx(self, name: str):
        prof = self._profiler
        if prof is not None and getattr(prof, "enabled", False):
            return prof.rf(name)
        return contextlib.nullcontext()

    def _do_show(self, img: Union[torch.Tensor, np.ndarray]):
        with self._prof_ctx("shower._do_show"), cuda_stream_scope(self._stream):
            if self._use_tk and self._tk_displayer is None:
                # root = get_tk_root()
                # self._tk_displayer = ImageDisplayer(root)
                assert False

            if img.ndim == 3:
                if isinstance(img, torch.Tensor | StreamTensorBase):
                    img = img.unsqueeze(0)
                elif isinstance(img, np.ndarray):
                    img = np.expand_dims(img, axis=0)

            img = unwrap_tensor(img)
            for s_img in img:
                if self._use_tk:
                    self._tk_displayer.display(s_img)
                else:
                    if self._cv2_has_opengl_support and s_img.device.type == "cuda":
                        # This doesn't work last time I checked
                        show_gpu_tensor(label=self._label, tensor=s_img, wait=False)
                    else:
                        if self._allow_gpu_gl and s_img.device.type == "cuda":
                            s_img = make_visible_image(
                                s_img, enable_resizing=self._show_scaled, force_numpy=False
                            )
                            self._stream.synchronize()
                            show_cuda_tensor("Stitched Image", s_img, False, None)
                        else:
                            cv2.imshow(
                                self._label,
                                make_visible_image(
                                    s_img, enable_resizing=self._show_scaled, force_numpy=True
                                ),
                            )
                            cv2.waitKey(1)

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None

    def _worker(self):
        last_frame = None
        next_frame_time = time.time()
        frame_interval = 1.0 / self._fps if self._fps is not None else None

        with cuda_stream_scope(self._stream):
            while True:
                if self._fps is None:
                    img = self._q.get()
                    if img is None:
                        break
                    self._do_show(img=img)
                else:
                    current_time = time.time()
                    sleep_duration = next_frame_time - current_time

                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                    # Update the next frame time
                    next_frame_time += frame_interval

                    # Determine which frame to show
                    frame_to_show = last_frame

                    while self._q.qsize() != 0:
                        potential_frame = self._q.get()
                        if time.time() >= next_frame_time - frame_interval:
                            frame_to_show = potential_frame

                    # If we have a frame to show, display it
                    if frame_to_show is not None:
                        self._do_show(frame_to_show)
                        last_frame = frame_to_show

    def _ensure_stream(self, device: torch.device):
        if self._stream is None and device.type == "cuda":
            # We give our dipslay stream high priority to reduce its dependency on other
            # computations and try to show the frame as soon as possible, in its (possibly) bad data state.
            self._stream = torch.cuda.Stream(device, priority=-1)

    def show(self, img: Union[torch.Tensor, np.ndarray, StreamTensorBase], clone: bool = False):
        self._iter += 1
        if self._iter % self._step != 0:
            return
        with self._prof_ctx("shower.show"):
            if self._stream is None and img.device.type == "cuda":
                self._ensure_stream(img.device)
            if self._thread is not None:
                counter: int = 0
                while self._q.qsize() >= self._max_size:
                    # print("Too many items in Shower queue...")
                    time.sleep(0.01)
                    counter += 1
                    if counter % 20 == 0:
                        self._logger.info("Too many items in Shower queue...")
                if self._cache_on_cpu and not isinstance(img, np.ndarray):
                    img = img.cpu()
                if self._fps is None or img.ndim == 3:
                    if not self._cache_on_cpu:
                        # if self._stream is not None:
                        #     self._stream.wait_stream(torch.cuda.current_stream(img.device))
                        # torch.cuda.current_stream(img.device).synchronize()
                        with cuda_stream_scope(self._stream):
                            img = unwrap_tensor(img)
                        if clone:
                            img = img.clone()
                        img = wrap_tensor(img)
                    self._q.put(img)
                else:
                    assert img.ndim == 4
                    for s_img in img:
                        assert False  # stream issues here sometimes if it cant be strided maybe?
                        self._q.put(s_img)
