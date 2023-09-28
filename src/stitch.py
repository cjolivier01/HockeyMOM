"""
Experiments in stitching
"""
import os
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import threading

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


def get_vid_metadata(vid):
    vid_cap = cv2.VideoCapture(vid.as_posix())
    return vid_cap.get(7), vid_cap.get(5), vid_cap.get(7) / vid_cap.get(5)


def create_wide_df(df):
    df["event"] = df["event"].shift(-1)
    df["event_attributes"] = df["event_attributes"].shift(-1)
    df["start_time"] = df["time"]
    df["event_time"] = df["time"].shift(-1)
    df["end_time"] = df["time"].shift(-2)
    df = df.query('event not in ["start", "end"]')
    df = df.drop(["time"], axis=1)
    return df


def create_new_interval_vid_df(vid, df, interval=2):
    _, _, length = get_vid_metadata(vid)

    start_times = np.arange(0, length, interval)
    end_times = np.arange(interval - 0.04, length + interval - 0.04, interval)

    if end_times[-1] > length:
        end_times[-1] = length

    vid_id = vid.name[:-4]
    vid_df = pd.DataFrame(
        {
            "vid_id": vid_id,
            "start_time": start_times,
            "end_time": end_times,
            "event": "background",
        }
    )

    old_vid_df = df.query("video_id == @vid_id")

    for event_time, event in zip(old_vid_df.event_time, old_vid_df.event):
        vid_df.loc[np.argmax(event_time < start_times), "event"] = event

    return vid_df


def create_interval_df(vids, df, interval):
    vid_dfs = []
    for vid in vids:
        vid_dfs.append(create_new_interval_vid_df(vid, df, interval))
    return pd.concat(vid_dfs, axis=0).reset_index(drop=True)


def get_frames_interval(vid_path, start_time, end_time, frame_transform=None):
    vid_cap = cv2.VideoCapture(vid_path.as_posix())
    fps = vid_cap.get(5)
    frames = vid_cap.get(7)
    length = vid_cap.get(7) / vid_cap.get(5)

    start_time = round(start_time / 0.04) * 0.04
    end_time = round(end_time / 0.04) * 0.04

    start_frame = timestamp_to_frame[start_time]
    end_frame = timestamp_to_frame[end_time]

    frames = []

    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = vid_cap.read()
    frame = torch.tensor(frame)

    if frame_transform != None:
        frame = frame_transform(frame)

    frames.append(frame)

    i = start_frame + 1

    while i <= end_frame:
        ret, frame = vid_cap.read()
        frame = torch.tensor(frame)

        if frame_transform != None:
            frame = frame_transform(frame)

        frames.append(frame)
        i += 1

    return torch.stack(frames, 0)


def eval(video_number: int, callback_fn: callable = None):
    input_dir = os.path.join(os.environ["HOME"], "Videos")
    # left_file = os.path.join(input_dir, f"left-{video_number}.mp4")
    # right_file = os.path.join(input_dir, f"right-{video_number}.mp4")
    left_file = os.path.join(input_dir, f"left.mp4")
    right_file = os.path.join(input_dir, f"right.mp4")
    return stitch_images(
        left_file=left_file,
        right_file=right_file,
        video_number=video_number,
        callback_fn=callback_fn,
    )


def expand_image_width(input_image, new_width: int):
    # Get the dimensions of the input image
    height, width, channels = input_image.shape

    # Create a new image with twice the width
    output_image = np.zeros((height, int(new_width), channels), dtype=np.uint8)

    # Calculate the padding on both sides
    # padding_left = (new_width - width) // 2
    # padding_right = new_width - width - padding_left

    # Copy the input image to the center of the new image
    # output_image[:, padding_left:padding_left + width, :] = input_image

    output_image[:, 0:width, :] = input_image
    return output_image


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
    # Create a mask for fully opaque pixels
    mask = image[:, :, 3] == 255

    # Find the coordinates of fully opaque pixels
    y, x = np.where(mask)

    # Find the largest ROI by calculating the bounding box
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    # Calculate the width and height of the largest ROI
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    print(f"Box: ({x_min}, {y_min}) -> ({x_max}, {y_max})")


PROCESSED_COUNT = 0


def feed_next_frame(
    video1: cv2.VideoCapture,
    video2: cv2.VideoCapture,
    data_loader: core.StitchingDataLoader,
    frame_id: int,
) -> bool:
    ret1, img1 = video1.read()
    if not ret1:
        return False
    # Read the corresponding frame from the second video
    ret2, img2 = video2.read()
    if not ret2:
        return False
    # print(f"Pushing frame {current_frame_id}")
    core.add_to_stitching_data_loader(data_loader, frame_id, img1, img2)
    return True


def get_next_frame(data_loader: core.StitchingDataLoader, frame_id: int):
    stitched_frame = core.get_stitched_frame_from_data_loader(data_loader, frame_id)
    if stitched_frame is None:
        raise StopIteration
    return stitched_frame


def frame_feeder_worker(
    video1: cv2.VideoCapture,
    video2: cv2.VideoCapture,
    data_loader: core.StitchingDataLoader,
    current_frame_id: int,
    max_frames: int,
):
    frame_count = 0
    while frame_count < max_frames:
        while frame_count - PROCESSED_COUNT > 50:
            time.sleep(0.1)
        if not feed_next_frame(
            video1=video1,
            video2=video2,
            data_loader=data_loader,
            frame_id=current_frame_id,
        ):
            break
        frame_count += 1
        current_frame_id += 1
    print("Feeder thread exiting")


def get_next_frame_sync(
    video1: cv2.VideoCapture,
    video2: cv2.VideoCapture,
    data_loader: core.StitchingDataLoader,
    frame_id: int,
):
    if not feed_next_frame(
        video1=video1, video2=video2, data_loader=data_loader, frame_id=frame_id
    ):
        raise StopIteration
    return get_next_frame(data_loader, frame_id)


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

    data_loader = core.StitchingDataLoader(0, pto_project_file, 25, 50, 50)

    # start_frame_number = 2000
    start_frame_number = 0
    # frame_step = 1200
    frame_id = start_frame_number
    # frame_step = 1
    max_frames = 20000
    skip_timing_frame_count = 50

    video1 = cv2.VideoCapture(f"{vid_dir}/left.mp4")
    video2 = cv2.VideoCapture(f"{vid_dir}/right.mp4")

    video1.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number + 217)
    video2.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number + 0)

    # Process the first frame in order to determine the output size
    # first_stitched_frame = get_next_frame_sync(
    #     video1=video1, video2=video2, data_loader=data_loader, frame_id=frame_id
    # )
    # final_video_size = (first_stitched_frame.shape[1], first_stitched_frame.shape[0])

    write_output_video = True
    output_video = None

    # fps = video1.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # output_video = cv2.VideoWriter(
    #     # filename=self._save_dir + "/../tracking_output.mov",
    #     filename="./stitched_output.avi",
    #     fourcc=fourcc,
    #     fps=fps,
    #     frameSize=final_video_size,
    #     isColor=True,
    # )
    # assert output_video.isOpened()
    # output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)
    # output_video.write(first_stitched_frame)

    # EARLY EXIT
    # output_video.release()
    # exit(0)

    def _maybe_write_output(output_img):
        nonlocal write_output_video
        if write_output_video:
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

            output_video.write(output_img)

    total_num_frames = min(
        int(video1.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(video2.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    max_frames = min(total_num_frames - start_frame_number, max_frames)
    assert max_frames > 0

    feeder_thread = threading.Thread(
        target=frame_feeder_worker,
        args=(video1, video2, data_loader, start_frame_number, max_frames),
    )
    feeder_thread.start()

    timer = Timer()

    frame_count = 0
    duration = 0
    start = None
    while frame_count < max_frames:
        if frame_count == skip_timing_frame_count:
            start = time.time()
        if frame_count and frame_count % 50 == 0:
            print(f"{frame_count} frames...")
        PROCESSED_COUNT += 1
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

            stitched_frame = core.get_stitched_frame_from_data_loader(
                data_loader, frame_id
            )

            if (frame_count + 1) % 20 == 0:
                timer.toc()
                logger.info(
                    "Stitching frame {} ({:.2f} fps)".format(
                        frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )

            # assert stitched_frame.shape[0] == final_video_size[1]
            # assert stitched_frame.shape[1] == final_video_size[0]
            # duration = time.time() - start
            # print(f"Got results in {duration} seconds")
            # if frame_count % 10 == 0:
            # cv2.imshow('Stitched', stitched_frame)
            # cv2.waitKey(0)
            _maybe_write_output(stitched_frame)

            if output_video is not None:
                output_video.write(stitched_frame)

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
