"""
Experiments in stitching
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import threading
import multiprocessing

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import tifffile

from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse
from lib.tracking_utils.log import logger

from hockeymom import core


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

        self.duration = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.duration = 0.0


def get_tiff_tag_value(tiff_tag):
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator) / denominator


def get_image_geo_position(tiff_image_file: str):
    xpos, ypos = 0, 0
    with tifffile.TiffFile(tiff_image_file) as tif:
        tags = tif.pages[0].tags
        # Access the TIFFTAG_XPOSITION
        x_position = get_tiff_tag_value(tags.get("XPosition"))
        y_position = get_tiff_tag_value(tags.get("YPosition"))
        x_resolution = get_tiff_tag_value(tags.get("XResolution"))
        y_resolution = get_tiff_tag_value(tags.get("YResolution"))
        xpos = int(x_position * x_resolution + 0.5)
        ypos = int(y_position * y_resolution + 0.5)
        print(f"x={xpos}, y={ypos}")
    return xpos, ypos


def build_stitching_project(project_file_path: str, skip_if_exists: bool = True):
    pass


def find_roi(image):
    binary_mask = image[:, :, 3] != 0

    # This may be inverted!
    distance_y = np.argmax(binary_mask, axis=0)
    distance_y = distance_y * (distance_y < image.shape[0] // 2)
    top = np.max(distance_y)

    distance_x = np.argmax(binary_mask, axis=1)
    distance_x = distance_x * (distance_x < image.shape[1] // 2)
    left = np.max(distance_x)

    flipped_image_ud = np.flipud(binary_mask)
    distance_y = np.argmax(flipped_image_ud, axis=0)
    distance_y = distance_y * (distance_y < image.shape[0] // 2)
    bottom = image.shape[0] - 1 - np.max(distance_y)

    flipped_image_lr = np.flipud(binary_mask)
    distance_x = np.argmax(flipped_image_lr, axis=1)
    distance_x = distance_x * (distance_x < image.shape[1] // 2)
    right = image.shape[1] - 1 - np.max(distance_x)

    print(f"Box: ({left}, {top}) -> ({right}, {bottom})")


PROCESSED_COUNT = 0


# def feed_next_frame(
#     video1: cv2.VideoCapture,
#     video2: cv2.VideoCapture,
#     stitching_data_loader_object: core.StitchingDataLoader,
#     frame_id: int,
# ) -> bool:
#     ret1, img1 = video1.read()
#     if not ret1:
#         return False
#     # Read the corresponding frame from the second video
#     ret2, img2 = video2.read()
#     if not ret2:
#         return False

#     # img1 = cv2.resize(img1, (img1.shape[1//4, img1.shape[1]//4]))
#     # img2 = cv2.resize(img2, (img2.shape[1//4, img2.shape[1]//4]))

#     # print(f"Pushing frame {current_frame_id}")
#     core.add_to_stitching_data_loader(
#         stitching_data_loader_object, frame_id, img1, img2
#     )
#     return True


# def get_next_frame(
#     stitching_data_loader_object: core.StitchingDataLoader, frame_id: int
# ):
#     stitched_frame = core.get_stitched_frame_from_data_loader(
#         stitching_data_loader_object, frame_id
#     )
#     if stitched_frame is None:
#         raise StopIteration
#     stitched_frame = prepare_frame_for_video(
#         stitched_frame,
#         image_roi=None,
#     )
#     return stitched_frame


# def frame_feeder_worker(
#     video1: cv2.VideoCapture,
#     video2: cv2.VideoCapture,
#     stitching_data_loader_object: core.StitchingDataLoader,
#     current_frame_id: int,
#     max_frames: int,
# ):
#     frame_count = 0
#     while frame_count < max_frames:
#         while frame_count - PROCESSED_COUNT > 100:
#             time.sleep(0.001)
#         if not feed_next_frame(
#             video1=video1,
#             video2=video2,
#             stitching_data_loader_object=stitching_data_loader_object,
#             frame_id=current_frame_id,
#         ):
#             break

#         # print("Skipping 200 frames...")
#         # video1.set(cv2.CAP_PROP_POS_FRAMES, video1.get(cv2.CAP_PROP_POS_FRAMES) + 200)
#         # video2.set(cv2.CAP_PROP_POS_FRAMES, video2.get(cv2.CAP_PROP_POS_FRAMES) + 200)

#         frame_count += 1
#         current_frame_id += 1
#     print("Feeder thread exiting")


# def start_feeder_thread(
#     video1: cv2.VideoCapture,
#     video2: cv2.VideoCapture,
#     stitching_data_loader_object: core.StitchingDataLoader,
#     start_frame_number: int,
#     max_frames: int,
# ):
#     feeder_thread = threading.Thread(
#         target=frame_feeder_worker,
#         args=(
#             video1,
#             video2,
#             stitching_data_loader_object,
#             start_frame_number,
#             max_frames,
#         ),
#     )
#     feeder_thread.start()
#     return feeder_thread


# # def stop_feeder_thread(feeder_thread):
# #     feeder_thread.stop = True


# def get_next_frame_sync(
#     video1: cv2.VideoCapture,
#     video2: cv2.VideoCapture,
#     stitching_data_loader_object: core.StitchingDataLoader,
#     frame_id: int,
# ):
#     if not feed_next_frame(
#         video1=video1,
#         video2=video2,
#         stitching_data_loader_object=stitching_data_loader_object,
#         frame_id=frame_id,
#     ):
#         raise StopIteration
#     return get_next_frame(stitching_data_loader_object, frame_id)


# def prepare_frame_for_video(image, image_roi):
#     if not image_roi:
#         if image.shape[2] == 4:
#             image = image[:, :, :3]
#     else:
#         image = image[image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3]
#     return image


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
        max_input_queue_size: int = 50,
        remap_thread_count: int = 10,
        blend_thread_count: int = 10,
        max_frames: int = None,
    ):
        assert max_input_queue_size > 0
        self._start_frame_number = start_frame_number
        self._output_stitched_video_file = output_stitched_video_file
        self._output_video = None
        self._video_1_offset_frame = video_1_offset_frame
        self._video_2_offset_frame = video_2_offset_frame
        self._video_file_1 = video_file_1
        self._video_file_2 = video_file_2
        self._pto_project_file = pto_project_file
        self._max_input_queue_size = max_input_queue_size
        self._remap_thread_count = remap_thread_count
        self._blend_thread_count = blend_thread_count
        self._max_frames = max_frames
        self._to_worker_queue = multiprocessing.Queue()
        self._from_worker_queue = multiprocessing.Queue()
        self._open = False
        self._current_frame = start_frame_number
        self._last_requested_frame = None
        self._feeder_thread = None

    def _open_videos(self):
        self._video1 = cv2.VideoCapture(self._video_file_1)
        self._video2 = cv2.VideoCapture(self._video_file_2)
        if self._start_frame_number or self._video_1_offset_frame:
            self._video1.set(
                cv2.CAP_PROP_POS_FRAMES,
                self._start_frame_number + self._video_1_offset_frame,
            )
        if self._start_frame_number or self._video_2_offset_frame:
            self._video2.set(
                cv2.CAP_PROP_POS_FRAMES,
                self._start_frame_number + self._video_2_offset_frame,
            )
        self._total_num_frames = min(
            int(self._video1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self._video2.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._stitcher = core.StitchingDataLoader(
            0,
            self._pto_project_file,
            self._max_input_queue_size,
            self._remap_thread_count,
            self._blend_thread_count,
        )
        self._start_feeder_thread()
        self._open = True

    def close(self):
        self._video1.release()
        self._video2.release()
        if self._output_video is not None:
            self._output_video.release()
        self._open = False

    def _maybe_write_output(self, output_img):
        if self._output_stitched_video_file:
            if self._output_video is None:
                fps = self._video1.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                final_video_size = (output_img.shape[1], output_img.shape[0])
                self._output_video = cv2.VideoWriter(
                    filename=self._output_stitched_video_file,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=final_video_size,
                    isColor=True,
                )
                assert self._output_video.isOpened()
                self._output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)

            self._output_video.write(output_img)

    def feed_next_frame(
        self,
    ) -> bool:
        frame_id = self._to_worker_queue.get()
        if frame_id is None:
            raise StopIteration
        frame_id = int(frame_id)
        ret1, img1 = self._video1.read()
        if not ret1:
            return False
        # Read the corresponding frame from the second video
        ret2, img2 = self._video2.read()
        if not ret2:
            return False
        print(f"Pushing frame {frame_id}")
        core.add_to_stitching_data_loader(self._stitcher, frame_id, img1, img2)
        return True

    def get_next_frame(self, frame_id: int):
        stitched_frame = core.get_stitched_frame_from_data_loader(
            self._stitcher, frame_id
        )
        if stitched_frame is None:
            raise StopIteration
        stitched_frame = self.prepare_frame_for_video(
            stitched_frame,
            image_roi=None,
        )
        return stitched_frame

    def frame_feeder_worker(
        self,
        max_frames: int,
    ):
        frame_count = 0
        while not max_frames or frame_count < max_frames:
            if not self.feed_next_frame():
                break
            frame_count += 1
        print("Feeder thread exiting")

    def _start_feeder_thread(self):
        self._feeder_thread = threading.Thread(
            target=self.frame_feeder_worker,
            args=(
                self._max_frames,
            ),
        )
        self._feeder_thread.start()
        for i in range(self._max_input_queue_size):
            req_frame = self._current_frame + i
            if not self._max_frames or req_frame < self._start_frame_number + self._max_frames:
                self._to_worker_queue.put(req_frame)
                self._last_requested_frame = req_frame

    # def stop_feeder_thread(feeder_thread):
    #     feeder_thread.stop = True

    def prepare_frame_for_video(self, image, image_roi):
        if not image_roi:
            if image.shape[2] == 4:
                image = image[:, :, :3]
        else:
            image = image[image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3]
        return image

    def __iter__(self):
        if not self._open:
            self._open_videos()
        return self

    def __next__(self):
        stitched_frame = self.get_next_frame(self._current_frame)
        self._current_frame += 1
        self._last_requested_frame += 1
        self._to_worker_queue.put(self._last_requested_frame)
        self._maybe_write_output(stitched_frame)
        return stitched_frame

    def __len__(self):
        return self._total_num_frames


# def create_data_loader(
#     pto_project_file: str,
#     start_frame_id: int,
#     max_input_queue_size: int,
#     remap_thread_count: int,
#     blend_thread_count: int,
# ):
#     to_fork_queue


def stitch_videos():
    vid_dir = os.path.join(os.environ["HOME"], "Videos")
    # orig_files_left = [
    #     f"{vid_dir}/images/left.png",
    #     f"{vid_dir}/images/left-45min.png",
    # ]

    # orig_files_right = [
    #     f"{vid_dir}/images/right.png",
    #     f"{vid_dir}/images/right-45min.png",
    # ]

    global PROCESSED_COUNT

    # PTO Project File
    pto_project_file = f"{vid_dir}/my_project.pto"

    build_stitching_project(pto_project_file)
    nona = core.HmNona(pto_project_file)

    print("Creating data loader...")
    stitching_data_loader_object = core.StitchingDataLoader(
        0, pto_project_file, 50, 2, 2
    )
    print("Data loader created.")

    # start_frame_number = 2000
    start_frame_number = 0
    # frame_step = 1200
    frame_id = start_frame_number
    # frame_step = 1
    max_frames = 300

    skip_timing_frame_count = 50

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{vid_dir}/left.mp4",
        video_file_2=f"{vid_dir}/right.mp4",
        pto_project_file=pto_project_file,
        video_1_offset_frame=217,
        video_2_offset_frame=0,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
    )

    for i, stitched_image in enumerate(data_loader):
        print(i)
    sys.exit(0)

    video1 = cv2.VideoCapture(f"{vid_dir}/left.mp4")
    video2 = cv2.VideoCapture(f"{vid_dir}/right.mp4")

    video1.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number + 217)
    video2.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number + 0)

    # Process the first frame in order to determine the output size
    # first_stitched_frame = get_next_frame_sync(
    #     video1=video1, video2=video2, stitching_data_loader_object=stitching_data_loader_object, frame_id=frame_id
    # )
    # image_roi = find_roi(first_stitched_frame)
    # first_stitched_frame = prepare_frame_for_video(first_stitched_frame, image_roi=image_roi)
    # #_, first_stitched_frame = video1.read()

    # final_video_size = (first_stitched_frame.shape[1], first_stitched_frame.shape[0])

    output_video = None

    # fps = video1.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # output_video = cv2.VideoWriter(
    #     filename="./stitched_output.avi",
    #     fourcc=fourcc,
    #     fps=fps,
    #     frameSize=final_video_size,
    #     isColor=True,
    # )
    # assert output_video.isOpened()
    # output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)

    # output_video.write(first_stitched_frame)
    # assert output_video.isOpened()

    # for i in range(10):
    #     print(f"writing frame {i+1}")
    #     # frame = get_next_frame_sync(
    #     #     video1=video1, video2=video2, stitching_data_loader_object=stitching_data_loader_object, frame_id=frame_id
    #     # )
    #     #output_video.write(frame)
    #     output_video.write(first_stitched_frame)
    #     assert output_video.isOpened()
    # output_video.release()
    # sys.exit(0)

    # EARLY EXIT
    # output_video.release()
    # exit(0)

    def _maybe_write_output(output_img):
        nonlocal output_stitched_video_file
        if output_stitched_video_file:
            nonlocal output_video
            if output_video is None:
                nonlocal video1
                fps = video1.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                final_video_size = (output_img.shape[1], output_img.shape[0])
                output_video = cv2.VideoWriter(
                    filename="./stitched_output.avi",
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=final_video_size,
                    isColor=True,
                )
                assert output_video.isOpened()
                output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)

            output_video.write(output_img)

    total_num_frames = min(
        int(video1.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(video2.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    max_frames = min(total_num_frames - start_frame_number, max_frames)
    assert max_frames > 0

    feeder_thread = start_feeder_thread(
        video1=video1,
        video2=video2,
        stitching_data_loader_object=stitching_data_loader_object,
        start_frame_number=start_frame_number,
        max_frames=max_frames,
    )

    timer = Timer()

    frame_count = 0
    duration = 0
    start = None
    while frame_count < max_frames:
        if frame_count == skip_timing_frame_count:
            start = time.time()
            timer = Timer()
        if frame_count and frame_count % 50 == 0:
            print(f"{frame_count} frames...")
        # ret1, img1 = video1.read()
        # if not ret1:
        #     break
        # # Read the corresponding frame from the second video
        # ret2, img2 = video2.read()
        # if not ret2:
        #     break

        # img1 = cv2.imread(orig_files_left[i % len(orig_files_left)])
        # img2 = cv2.imread(orig_files_right[i % len(orig_files_left)])
        # assert img1 is not None and img2 is not None
        # cv2.imshow('Nona image left', img1)
        # cv2.waitKey(0)
        # cv2.imshow('Nona image right', img2)
        # cv2.waitKey(0)
        # start = time.time()
        if True:
            if frame_count > 1:
                timer.tic()

            stitched_frame = get_next_frame(stitching_data_loader_object, frame_id)
            PROCESSED_COUNT += 1

            # assert stitched_frame.shape[0] == final_video_size[1]
            # assert stitched_frame.shape[1] == final_video_size[0]
            # duration = time.time() - start
            # print(f"Got results in {duration} seconds")
            # if frame_count % 10 == 0:
            # cv2.imshow('Stitched', stitched_frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            _maybe_write_output(stitched_frame)

            # if output_video is not None:
            #     output_video.write(stitched_frame)

            if (frame_count + 1) % 20 == 0:
                timer.toc()
                logger.info(
                    "Stitching frame {} ({:.2f} fps)".format(
                        frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )
                timer.clear()
        elif True:
            result = core.nona_process_images(nona, img1, img2)
            duration = time.time() - start
            print(f"Got results in {duration} seconds")
            # cv2.imshow('Nona image left', result[0])
            # cv2.waitKey(0)
            # cv2.imshow('Nona image right', result[1])
            # cv2.waitKey(0)
        else:
            result = core.stitch_images(nona, img1, img2)
            duration = time.time() - start
            print(f"Got results in {duration} seconds")
            # cv2.imshow('Stitched Image', result)
            # cv2.waitKey(0)

        frame_id += 1
        frame_count += 1

        # if frame_step > 1:
        #     video1.set(
        #         cv2.CAP_PROP_POS_FRAMES,
        #         video1.get(cv2.CAP_PROP_POS_FRAMES) + frame_step - 1,
        #     )
        #     video2.set(
        #         cv2.CAP_PROP_POS_FRAMES,
        #         video2.get(cv2.CAP_PROP_POS_FRAMES) + frame_step - 1,
        #     )
    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count - skip_timing_frame_count} frames in {duration} seconds ({(frame_count - skip_timing_frame_count)/duration} fps)"
        )
    if output_video is not None:
        output_video.release()
    # files_left = [
    #     f"{vid_dir}/my_project0000.tif",
    #     f"{vid_dir}/my_project-20000.tif",
    # ]
    # files_right = [
    #     f"{vid_dir}/my_project0001.tif",
    #     f"{vid_dir}/my_project-20001.tif",
    # ]

    # xpos_1, ypos_1 = get_image_geo_position(files_left[0])
    # xpos_2, ypos_2 = get_image_geo_position(files_right[0])

    # for i in range(len(files_left)):
    #     A = cv2.imread(files_left[i])
    #     B = cv2.imread(files_right[i])

    #     img = core.emblend_images(A, B, [xpos_1, ypos_1], [xpos_2, ypos_2])
    #     # cv2.imshow('Panoramic blended image', img)
    #     # cv2.waitKey(0)

    # cv2.destroyAllWindows()


def main():
    stitch_videos()


if __name__ == "__main__":
    main()
    print("Done.")
