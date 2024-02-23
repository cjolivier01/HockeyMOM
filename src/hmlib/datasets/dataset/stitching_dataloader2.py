"""
Experiments in stitching
"""

import os
import cv2
import threading
import argparse
import torch
import traceback
import numpy as np
from typing import List, Union

from pathlib import Path

from hockeymom import core
from yolox.data.dataloading import get_yolox_datadir

from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.stitching.synchronize import (
    configure_video_stitching,
)
from hmlib.stitching.blender import create_stitcher
from hmlib.stitching.laplacian_blend import show_image
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.utils.gpu import StreamTensor, StreamCheckpoint
from hmlib.video_out import VideoOutput, ImageProcData, optional_with
from hmlib.utils.image import (
    make_channels_last,
    make_channels_first,
    image_height,
    image_width,
)


def _get_dir_name(path):
    if os.path.isdir(path):
        return path
    return Path(path).parent


from hmlib.stitching.stitch_worker import (
    # StitchingWorker,
    create_queue,
    safe_put_queue,
    INFO,
    _LARGE_NUMBER_OF_FRAMES,
)


def to_tensor(tensor: Union[torch.Tensor, StreamTensor]):
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, StreamTensor):
        return tensor.get()
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    else:
        assert False


def sync_required(tensor: Union[torch.Tensor, StreamTensor]):
    if isinstance(tensor, (torch.Tensor, np.ndartray)):
        return False
    if tensor.owns_stream:
        return False
    return True


def distribute_items_detailed(total_item_count, worker_count):
    base_items_per_worker = total_item_count // worker_count
    remainder = total_item_count % worker_count

    distribution = []
    for i in range(worker_count):
        if i < remainder:
            distribution.append(base_items_per_worker + 1)
        else:
            distribution.append(base_items_per_worker)

    return distribution


# def _get_cuda_device():
#     # Use last device
#     if int(os.environ.get("HM_NO_CUDA_STITCH_ENCODER", "0")):
#         return torch.device("cpu")
#     return torch.device("cuda", torch.cuda.device_count() - 1)


class MultiDataLoaderWrapper:
    def __init__(self, dataloaders: List[MOTLoadVideoWithOrig]):
        self._dataloaders = dataloaders
        self._iters = []
        self._len = None

    def __iter__(self):
        self._iters = []
        for dl in self._dataloaders:
            self._iters.append(iter(dl))
        return self

    def close(self):
        for dl in self._dataloaders:
            dl.close()

    def __next__(self):
        result = []
        for it in self._iters:
            result.append(next(it))
        if not result:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return result

    def __len__(self):
        if self._len is None:
            self._len = -1
            for dl in self._dataloaders:
                this_len = len(dl)
                self._len = this_len if self._len is None else min(self._len, this_len)


def as_torch_device(device):
    if isinstance(device, str):
        return torch.device(device)
    return device


##
#   _____ _   _  _        _     _____        _                     _
#  / ____| | (_)| |      | |   |  __ \      | |                   | |
# | (___ | |_ _ | |_  ___| |__ | |  | | __ _| |_  __ _  ___   ___ | |_
#  \___ \| __| || __|/ __| '_ \| |  | |/ _` | __|/ _` |/ __| / _ \| __|
#  ____) | |_| || |_| (__| | | | |__| | (_| | |_| (_| |\__ \|  __/| |_
# |_____/ \__|_| \__|\___|_| |_|_____/ \__,_|\__|\__,_||___/ \___| \__|
#
#
class StitchDataset:
    def __init__(
        self,
        video_file_1: str,
        video_file_2: str,
        pto_project_file: str = None,
        video_1_offset_frame: int = None,
        video_2_offset_frame: int = None,
        output_stitched_video_file: str = None,
        start_frame_number: int = 0,
        max_input_queue_size: int = 2,
        remap_thread_count: int = 10,
        blend_thread_count: int = 10,
        batch_size: int = 1,
        max_frames: int = None,
        auto_configure: bool = True,
        num_workers: int = 1,
        fork_workers: bool = False,
        image_roi: List[int] = None,
        encoder_device: torch.device = torch.device("cpu"),
        blend_mode: str = "laplacian",
        remapping_device: torch.device = None,
        remap_on_async_stream: bool = False,
    ):
        assert max_input_queue_size > 0
        self._start_frame_number = start_frame_number
        self._batch_size = batch_size
        self._remapping_device = as_torch_device(remapping_device)
        self._remap_on_async_stream = remap_on_async_stream
        self._encoder_device = as_torch_device(encoder_device)
        self._output_stitched_video_file = output_stitched_video_file
        self._video_1_offset_frame = video_1_offset_frame
        self._video_2_offset_frame = video_2_offset_frame
        self._video_file_1 = video_file_1
        self._video_file_2 = video_file_2
        self._pto_project_file = pto_project_file
        self._max_input_queue_size = max_input_queue_size
        self._remap_thread_count = remap_thread_count
        self._blend_thread_count = blend_thread_count
        self._blend_mode = blend_mode
        self._max_frames = (
            max_frames if max_frames is not None else _LARGE_NUMBER_OF_FRAMES
        )
        self._to_coordinator_queue = create_queue(mp=False)
        self._from_coordinator_queue = create_queue(mp=False)
        self._current_frame = start_frame_number
        self._next_requested_frame = start_frame_number

        if self._remapping_device.type == "cuda":
            self._remapping_stream = torch.cuda.Stream(device=self._remapping_device)
        else:
            self._remapping_stream = None

        # Optimize the roi box
        if image_roi is not None:
            if isinstance(image_roi, (list, tuple)):
                if not any(item is not None for item in image_roi):
                    image_roi = None
        self._image_roi = image_roi

        self._fps = None
        self._bitrate = None
        self._auto_configure = auto_configure
        self._num_workers = num_workers
        self._stitching_workers = {}
        self._fork_workers = fork_workers
        self._batch_count = 0
        # Temporary until we get the middle-man (StitchingWorkersIterator)
        self._current_worker = 0
        self._ordering_queue = create_queue(mp=False)
        self._coordinator_thread = None

        self._next_frame_timer = Timer()
        self._next_frame_counter = 0

        self._next_timer = Timer()

        self._prepare_next_frame_timer = Timer()

        self._video_1_info = BasicVideoInfo(video_file_1)
        self._video_2_info = BasicVideoInfo(video_file_2)
        v1o = 0 if self._video_1_offset_frame is None else self._video_1_offset_frame
        v2o = 0 if self._video_2_offset_frame is None else self._video_2_offset_frame
        self._total_number_of_frames = int(
            min(
                self._video_1_info.frame_count - v1o,
                self._video_2_info.frame_count - v2o,
            )
        )
        self._stitcher = None
        self._video_output = None

    def __delete__(self):
        self.close()
        # for worker in self._stitching_workers.values():
        #     worker.close()

    @property
    def lfo(self):
        assert self._video_1_offset_frame is not None
        return self._video_1_offset_frame

    @property
    def rfo(self):
        assert self._video_2_offset_frame is not None
        return self._video_2_offset_frame

    def stitching_worker(self, worker_number: int):
        return self._stitching_workers[worker_number]

    def create_stitching_worker(
        self,
        rank: int,
        start_frame_number: int,
        frame_stride_count: int,
        max_frames: int,
        max_input_queue_size: int,
        remapping_device: torch.device,
        dataset_name: str = "crowdhuman",
    ):
        dataloaders = []
        dataloaders.append(
            MOTLoadVideoWithOrig(
                path=self._video_file_1,
                data_dir=os.path.join(get_yolox_datadir(), dataset_name),
                img_size=None,
                max_frames=max_frames,
                batch_size=self._batch_size,
                start_frame_number=start_frame_number,
                original_image_only=True,
                device=remapping_device,
                stream_tensors=True,
                # device=torch.device("cpu"),
                # scale_rgb_down=True,
            )
        )
        dataloaders.append(
            MOTLoadVideoWithOrig(
                path=self._video_file_2,
                data_dir=os.path.join(get_yolox_datadir(), dataset_name),
                img_size=None,
                max_frames=max_frames,
                batch_size=self._batch_size,
                start_frame_number=start_frame_number,
                original_image_only=True,
                device=remapping_device,
                stream_tensors=True,
                # device=torch.device("cpu"),
                # scale_rgb_down=True,
            )
        )
        stitching_worker = MultiDataLoaderWrapper(dataloaders=dataloaders)
        # stitching_worker = StitchingWorker(
        #     rank=rank,
        #     video_file_1=self._video_file_1,
        #     video_file_2=self._video_file_2,
        #     pto_project_file=self._pto_project_file,
        #     video_1_offset_frame=self._video_1_offset_frame,
        #     video_2_offset_frame=self._video_2_offset_frame,
        #     start_frame_number=start_frame_number,
        #     batch_size=self._batch_size,
        #     max_input_queue_size=max_input_queue_size,
        #     remap_thread_count=self._remap_thread_count,
        #     blend_thread_count=self._blend_thread_count,
        #     max_frames=max_frames,
        #     frame_stride_count=frame_stride_count,
        #     multiprocessingt_queue=self._fork_workers,
        #     remapping_device=remapping_device,
        #     blend_mode=self._blend_mode,
        # )
        # stitching_worker =
        return stitching_worker

    def configure_stitching(self):
        if not self._pto_project_file:
            dir_name = _get_dir_name(self._video_file_1)
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                dir_name,
                self._video_file_1,
                self._video_file_2,
                left_frame_offset=self._video_1_offset_frame,
                right_frame_offset=self._video_2_offset_frame,
            )
            self._video_1_offset_frame = lfo
            self._video_2_offset_frame = rfo

    def initialize(self):
        if self._auto_configure:
            self.configure_stitching()

    def _load_video_props(self):
        video1 = cv2.VideoCapture(self._video_file_1)
        if not video1.isOpened():
            raise AssertionError(f"Could not open video file: {self._video_file_1}")
        self._fps = video1.get(cv2.CAP_PROP_FPS)
        self._bitrate = video1.get(cv2.CAP_PROP_BITRATE)
        video1.release()

    @property
    def fps(self):
        if self._fps is None:
            self._load_video_props()
        return self._fps

    @property
    def bitrate(self):
        if self._bitrate is None:
            self._load_video_props()
        return self._bitrate

    def close(self):
        for stitching_worker in self._stitching_workers.values():
            stitching_worker.close()
        self._stop_coordinator_thread()
        self._stitching_workers.clear()
        if self._video_output is not None:
            self._video_output.stop()
            self._video_output = None

    def _prepare_next_frame(self, frame_id: int):
        # INFO(f"_prepare_next_frame( {frame_id} )")
        self._prepare_next_frame_timer.tic()

        stitching_worker = self._stitching_workers[self._current_worker]
        images = next(stitching_worker)

        imgs_1 = images[0][0]
        ids_1 = images[0][-1]

        imgs_2 = images[1][0]
        ids_2 = images[1][-1]

        assert ids_1 == ids_2

        assert isinstance(images, list)
        assert len(images) == 2

        def _prepare_image(img: torch.Tensor):
            img = make_channels_first(img)
            if img.device != self._remapping_device:
                img = img.to(self._remapping_device, non_blocking=True)
            if img.dtype not in [
                torch.float,
                torch.float16,
                torch.float32,
                torch.float64,
            ]:
                img = img.to(torch.float, non_blocking=True)
            return img

        stream = None
        if imgs_1.device.type == "cpu":
            stream = self._remapping_stream
        with optional_with(torch.cuda.stream(stream) if stream is not None else None):
            sinfo_1 = core.StitchImageInfo()
            sinfo_1.image = _prepare_image(imgs_1)
            sinfo_1.xy_pos = self._xy_pos_1

            sinfo_2 = core.StitchImageInfo()
            sinfo_2.image = _prepare_image(imgs_2)
            sinfo_2.xy_pos = self._xy_pos_2

            blended_stream_tensor = self._stitcher.forward(inputs=[sinfo_1, sinfo_2])
            if stream is not None:
                blended_stream_tensor = StreamCheckpoint(
                    tensor=blended_stream_tensor, stream=None
                )
            # if stream is not None:
            #     stream.synchronize()

        self._current_worker = (self._current_worker + 1) % len(self._stitching_workers)
        self._ordering_queue.put((ids_1, blended_stream_tensor))
        self._prepare_next_frame_timer.toc()

    def _start_coordinator_thread(self):
        assert self._coordinator_thread is None
        for _ in range(min(self._max_input_queue_size, self._max_frames)):
            # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._from_coordinator_queue.put(("ok", self._next_requested_frame))
            self._next_requested_frame += 1
        self._coordinator_thread = threading.Thread(
            name="StitchCoordinator",
            target=self._coordinator_thread_worker,
            args=(self._next_requested_frame,),
        )
        self._coordinator_thread.start()
        print("Cached request queue")

    def _stop_coordinator_thread(self):
        if self._coordinator_thread is not None:
            self._to_coordinator_queue.put("stop")
            self._coordinator_thread.join()
            self._coordinator_thread = None

    def _send_frame_to_video_out(self, frame_id: int, stitched_frame: torch.Tensor):
        if not self._output_stitched_video_file:
            return
        if self._video_output is None:
            args = argparse.Namespace()
            args.fixed_edge_rotation = False
            args.crop_output_image = False
            args.use_watermark = False
            args.show_image = False
            args.plot_frame_number = False
            self._video_output_size = torch.tensor(
                [image_width(stitched_frame), image_height(stitched_frame)],
                dtype=torch.int32,
            )
            self._video_output_box = torch.tensor(
                (0, 0, self._video_output_size[0], self._video_output_size[1]),
                dtype=torch.float32,
            )
            # Not doing anything fancy, so don't waste time copy to and from the GPU
            self._video_output = VideoOutput(
                args=args,
                output_video_path=self._output_stitched_video_file,
                output_frame_width=self._video_output_size[0],
                output_frame_height=self._video_output_size[1],
                fps=self.fps,
                device=self._encoder_device,
                # fourcc=(
                #     "hevc_nvenc"
                #     if str(self._encoder_device).startswith("cuda")
                #     else "XVID"
                # ),
                name="STITCH-OUT",
                simple_save=True,
            )
        image_proc_data = ImageProcData(
            frame_id=frame_id,
            img=torch.clamp(to_tensor(stitched_frame) / 255.0, min=0.0, max=255.0),
            current_box=self._video_output_box.clone(),
        )
        self._video_output.append(image_proc_data)

    def _coordinator_thread_worker(self, next_requested_frame, *args, **kwargs):
        try:
            #
            # Create the stitcher
            #
            assert self._stitcher is None
            self._stitcher, self._xy_pos_1, self._xy_pos_2 = create_stitcher(
                dir_name=os.path.dirname(self._video_file_1),
                batch_size=self._batch_size,
                device=self._remapping_device,
                remap_on_async_stream=self._remap_on_async_stream,
            )
            assert self._remapping_device.type != "cpu"
            self._stitcher.to(self._remapping_device)

            frame_count = 0
            while frame_count < self._max_frames:
                command = self._to_coordinator_queue.get()
                if isinstance(command, str) and command == "stop":
                    break
                frame_id = int(command)
                self._prepare_next_frame(frame_id)
                if next_requested_frame < self._start_frame_number + self._max_frames:
                    self._from_coordinator_queue.put(("ok", next_requested_frame))
                else:
                    # print(f"Not requesting frame {next_requested_frame}")
                    pass
                frame_count += 1
                next_requested_frame += 1
            safe_put_queue(self._from_coordinator_queue, StopIteration())
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            safe_put_queue(self._from_coordinator_queue, ex)

    @staticmethod
    def prepare_frame_for_video(image: np.array, image_roi: np.array):
        if not image_roi:
            if image.shape[-1] == 4:
                if len(image.shape) == 4:
                    image = make_channels_last(image)[:, :, :, :3]
                else:
                    image = make_channels_last(image)[:, :, :3]
        else:
            image_roi = fix_clip_box(
                image_roi, [image_height(image), image_width(image)]
            )
            if len(image.shape) == 4:
                image = make_channels_last(image)[
                    :, image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
            else:
                assert len(image.shape) == 3
                image = make_channels_last(image)[
                    image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
        return image

    def __iter__(self):
        if not self._stitching_workers:
            self.initialize()
            # Openend close to validate existance as well as get some stats, such as fps
            for worker_number in range(self._num_workers):
                max_for_worker = self._max_frames
                if max_for_worker is not None:
                    max_for_worker = distribute_items_detailed(
                        self._max_frames, self._num_workers
                    )[
                        worker_number
                    ]  # TODO: call just once
                self._stitching_workers[worker_number] = iter(
                    self.create_stitching_worker(
                        rank=worker_number,
                        start_frame_number=self._start_frame_number,
                        frame_stride_count=self._num_workers,
                        max_frames=max_for_worker,
                        max_input_queue_size=int(
                            self._max_input_queue_size / self._num_workers + 1
                        ),
                        remapping_device=self._remapping_device,
                    )
                )
                # self._stitching_workers[worker_number].start(fork=self._fork_workers)
            self._start_coordinator_thread()
        return self

    def get_next_frame(self, frame_id: int):
        self._next_frame_timer.tic()
        assert frame_id == self._current_frame
        # INFO(f"Dequeing frame id: {self._current_frame}...")
        # stitched_frame = self._ordering_queue.dequeue_key(self._current_frame)
        frame_id, stitched_frame = self._ordering_queue.get()

        # INFO(f"Locally dequeued frame id: {self._current_frame}")
        if (
            not self._max_frames
            or self._next_requested_frame < self._start_frame_number + self._max_frames
        ):
            # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._next_requested_frame += 1
        else:
            # We were pre-requesting future frames, but we're past the
            # frames we want, so don't ask for anymore and just return these
            # (running out what's in the queue)
            # INFO(
            #     f"Next frame {self._next_requested_frame} would be above the max allowed frames, so not queueing"
            # )
            pass

        if stitched_frame is not None:
            self._next_frame_timer.toc()
            self._next_frame_counter += 1

        return stitched_frame

    def __next__(self):
        # INFO(f"\nBEGIN next() self._from_coordinator_queue.get() {self._current_frame}")
        # print(f"self._from_coordinator_queue size: {self._from_coordinator_queue.qsize()}")
        self._next_timer.tic()
        status = self._from_coordinator_queue.get()
        # INFO(f"END next() self._from_coordinator_queue.get( {self._current_frame})\n")
        if isinstance(status, Exception):
            self.close()
            raise status
        else:
            status, frame_id = status
            assert status == "ok"
            # print(f"self._from_coordinator_queue.get() = {frame_id}, self._current_frame = {self._current_frame} ")
            assert frame_id == self._current_frame

        # self._next_timer.tic()
        stitched_frame = self.get_next_frame(frame_id=frame_id)

        self._batch_count += 1

        # show_image("stitched_frame", stitched_frame, wait=True)
        if stitched_frame is None:
            self.close()
            raise StopIteration()

        # Code doesn't handle strided channels efficiently
        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=self._image_roi,
        )

        self._send_frame_to_video_out(frame_id=frame_id, stitched_frame=stitched_frame)
        assert stitched_frame.ndim == 4
        # maybe nested batches can be some multiple of, so can remove this check if necessary
        assert self._batch_size == stitched_frame.shape[0]
        self._current_frame += stitched_frame.shape[0]
        self._next_timer.toc()

        if self._batch_count % 50 == 0:
            logger.info(
                "Stitching dataset __next__ wait speed {} ({:.2f} fps)".format(
                    self._current_frame,
                    self._batch_size * 1.0 / max(1e-5, self._next_timer.average_time),
                )
            )

        return stitched_frame

    def __len__(self):
        return self._total_number_of_frames


def is_none(val):
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def fix_clip_box(clip_box, hw: List[int]):
    if isinstance(clip_box, list):
        if is_none(clip_box[0]):
            clip_box[0] = 0
        if is_none(clip_box[1]):
            clip_box[1] = 0
        if is_none(clip_box[2]):
            clip_box[2] = hw[1]
        if is_none(clip_box[3]):
            clip_box[3] = hw[0]
        clip_box = np.array(clip_box, dtype=np.int64)
    return clip_box
