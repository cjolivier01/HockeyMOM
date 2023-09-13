import os
import torch
import torch.nn as nn

import random
import cv2

import pprint

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

import time


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


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # output audio from original
    output_audio_path = f"/tmp/output-audio-{uuid.uuid4().hex}.mp3"
    cmd_str = f'ffmpeg -i {original_video} -q:a 0 -map a {output_audio_path}'
    print(cmd_str)
    os.system(cmd_str)
    # attach audio to new video
    cmd_str = f'ffmpeg -i {soundless_video} -i {output_audio_path} -map 0:v -map 1:a -c:v copy -shortest {final_audio_video}'
    print(cmd_str)
    os.system(cmd_str)
    # delete temp audio
    if os.path.isfile(output_audio_path):
        os.unlink(output_audio_path)


def eval(video_number: int, callback_fn: callable = None):
    scale_down_images = True
    show_image = False
    skip_frame_count = 0
    filename_stitched = None
    filename_with_audio = None

    input_dir = os.path.join(os.environ["HOME"], "Downloads")
    left_file = os.path.join(input_dir, f"left-{video_number}.mp4")
    right_file = os.path.join(input_dir, f"right-{video_number}.mp4")
    vidcap_left = cv2.VideoCapture(left_file)
    vidcap_right = cv2.VideoCapture(right_file)

    fps = vidcap_left.get(cv2.CAP_PROP_FPS)
    frame_width = int(vidcap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vidcap_left.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS={fps}")
    print(f"Frame count={total_frames}")
    print(f"Input size: {frame_width} x {frame_height}")

    final_frame_width = frame_width * 2
    final_frame_height = frame_height

    if scale_down_images:
        final_frame_width = (frame_width * 2) // 2
        final_frame_height = frame_height // 2

    if callback_fn is None:
        filename_stitched = f"stitched-output-{video_number}.mov"
        out = cv2.VideoWriter(
            filename=filename_stitched,
            fourcc=cv2.VideoWriter_fourcc(*"XVID"),
            fps=fps,
            frameSize=(final_frame_width, final_frame_height),
            isColor=True,
        )
        assert out.isOpened()
        filename_with_audio = f"stitched-output-{video_number}-with-audio.mov"
    else:
        out = None

    r1, frame1 = vidcap_left.read()
    r2, frame2 = vidcap_right.read()

    frame_id = 0
    timer = Timer()
    while r1 and r2:
        # cv2.imwrite("frames/frame%d.png" % count, image)     # save frame as JPEG file
        timer.tic()
        r1, frame1 = vidcap_left.read()
        r2, frame2 = vidcap_right.read()

        if frame_id < skip_frame_count:
            frame_id += 1
            continue

        if final_frame_height != frame_height:
            frame1 = cv2.resize(frame1, (final_frame_width // 2, final_frame_height))
            frame2 = cv2.resize(frame2, (final_frame_width // 2, final_frame_height))

        # Concatenate the frames side-by-side
        combined_frame = cv2.hconcat([frame1, frame2])

        if show_image:
            cv2.imshow('Combined Image', combined_frame)
            cv2.waitKey(0)

        if out is not None:
            out.write(combined_frame)

        if callback_fn is not None:
            if not callback_fn(combined_frame, (final_frame_width, final_frame_height)):
                break

        timer.toc()
        if frame_id % 20 == 0:
            print(
                "Processing frame {}/{} ({:.2f} fps)".format(
                    frame_id, total_frames, 1.0 / max(1e-5, timer.average_time)
                )
            )
        frame_id += 1

    if out is not None:
        out.release()
    vidcap_left.release()
    vidcap_right.release()

    if show_image:
        cv2.destroyAllWindows()

    if filename_with_audio:
        copy_audio(left_file, filename_stitched, filename_with_audio)

def main():
    eval(video_number=2)


if __name__ == "__main__":
    main()
    print("Done.")
