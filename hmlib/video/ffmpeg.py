"""FFmpeg-based video metadata and subprocess helpers.

Provides :class:`BasicVideoInfo` and small utilities around ``ffprobe`` and
``ffmpeg`` used throughout the hmlib video pipeline.

@see @ref hmlib.video.video_stream "video_stream" for higher-level streaming.
"""

import ctypes
import os
from fractions import Fraction
import platform
import re
import signal
import subprocess
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hmlib.log import get_logger
from hmlib.utils.utils import classinstancememoize

libc = ctypes.CDLL("libc.so.6")


def preexec_fn():
    # Ensure the child process gets SIGTERM if the parent dies
    libc.prctl(1, signal.SIGTERM)


def _frame_rate_to_fraction(value) -> Fraction:
    """
    Convert an ffprobe-style frame rate (float or 'num/den' string) to a Fraction.

    Falls back to 0/1 on parse errors.
    """
    # Already a Fraction
    if isinstance(value, Fraction):
        return value
    # Numeric (int/float)
    try:
        if isinstance(value, (int, float)):
            return Fraction.from_float(float(value)).limit_denominator()
    except Exception:
        pass
    # String (e.g. "30000/1001" or "30")
    try:
        s = str(value).strip()
        if not s:
            return Fraction(0, 1)
        tokens = s.split("/")
        if len(tokens) == 1:
            return Fraction(int(tokens[0]), 1)
        if len(tokens) == 2:
            num = int(tokens[0])
            den = int(tokens[1])
            if den == 0:
                return Fraction(0, 1)
            return Fraction(num, den)
    except Exception:
        # Fall through to float-based parsing
        pass
    try:
        return Fraction.from_float(float(value)).limit_denominator()
    except Exception:
        return Fraction(0, 1)


@classinstancememoize
class BasicVideoInfo:
    def __init__(self, video_file: str, use_ffprobe: bool = True):
        assert isinstance(video_file, str)
        video_file = video_file.split(",")
        self._multiple = None
        if isinstance(video_file, list) and len(video_file) == 1:
            video_file = video_file[0]
        if isinstance(video_file, list):
            self._multiple = []
            for f in video_file:
                self._multiple.append(BasicVideoInfo(f, use_ffprobe=use_ffprobe))
            # Let's assume they're all the same
            firstone = self._multiple[0]
            self.fps = firstone.fps
            self.width = firstone.width
            self.height = firstone.height
            self.bit_rate = firstone.bit_rate
            self.codec = firstone.codec
            # accumulate/max
            self.duration = firstone.duration
            self.frame_count = firstone.frame_count
            for v in self._multiple[1:]:
                self.duration += v.duration
                self.frame_count += v.frame_count
                self.bit_rate = max(self.bit_rate, v.bit_rate)
        else:
            video_file = str(video_file)
            if use_ffprobe:
                probe = FFProbe(video_file)
                self._ffstream: Optional[FFStream] = None
                if not probe.video:
                    raise AssertionError(f"Unable to get video stream from file: {video_file}")
                elif len(probe.video) > 1:
                    # DJI camera has other weird streams, so just warn and use the first one
                    get_logger(__name__).warning(
                        "Found too many (%d) video streams in file: %s; using the first one only",
                        len(probe.video),
                        video_file,
                    )
                    # raise AssertionError(
                    #     f"Found too many ({len(probe.video)}) video streams in file: {video_file}"
                    # )
                self._ffstream = probe.video[0]
                # Duration in seconds is kept as float; fps is stored as Fraction
                self.duration = self._ffstream.durationSeconds()
                # Prefer the exact r_frame_rate string when available
                r_frame_rate = getattr(self._ffstream, "r_frame_rate", None)
                self.fps = _frame_rate_to_fraction(r_frame_rate)
                # Fall back to realFrameRate() if r_frame_rate was missing/invalid
                if self.fps == 0:
                    self.fps = _frame_rate_to_fraction(self._ffstream.realFrameRate())
                self.frame_count = int(self.duration * float(self.fps))
                sz = self._ffstream.frameSize()
                self.width = sz[0]
                self.height = sz[1]
                self.bit_rate = self._ffstream.bitrate()
                self.codec = self._ffstream.codecTag()
                # self.fourcc = fourcc_to_int(self._ffstream.codecTag())
            else:
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    raise AssertionError(f"Unable to open video file {video_file}")
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                self.fps = _frame_rate_to_fraction(cap_fps)
                # Avoid division by zero if FPS is unavailable
                self.duration = (
                    float("inf")
                    if float(self.fps) == 0.0
                    else self.frame_count / float(self.fps)
                )
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.bit_rate = int(cap.get(cv2.CAP_PROP_BITRATE) * 1000)
                self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                self.codec = "".join([chr((self.fourcc >> 8 * i) & 0xFF) for i in range(4)]).upper()
                cap.release()


def fourcc_to_int(fourcc):
    """
    Convert a FOURCC code to an integer.

    Args:
    - fourcc (str): A four-character string representing the FOURCC code.

    Returns:
    - int: The integer representation of the FOURCC code.
    """
    # Ensure the input is exactly 4 characters
    assert len(fourcc) == 4, f"FOURCC code must be 4 characters long ({fourcc})."

    # Calculate the integer value
    value = (
        (ord(fourcc[0]) << 0)
        | (ord(fourcc[1]) << 8)
        | (ord(fourcc[2]) << 16)
        | (ord(fourcc[3]) << 24)
    )

    return value


def duration_to_seconds(duration_str):
    # Split the duration string into hours, minutes, seconds, and nanoseconds
    hours, minutes, seconds_ns = duration_str.split(":")
    seconds, ns = seconds_ns.split(".")

    # Convert hours and minutes to seconds, and nanoseconds to seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ns) / 1e9

    return total_seconds


def print_ffmpeg_info():
    from torchio.utils import ffmpeg_utils

    logger = get_logger(__name__)
    logger.info("Library versions: %s", ffmpeg_utils.get_versions())
    logger.info("Build config: %s", ffmpeg_utils.get_build_config())
    logger.info(
        "Cuvid decoders: %s",
        [k for k in ffmpeg_utils.get_video_decoders().keys() if "cuvid" in k],
    )
    logger.info(
        "Nvenc encoders: %s",
        [k for k in ffmpeg_utils.get_video_encoders().keys() if "nvenc" in k],
    )


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # attach audio to new video
    cmd_str = (
        f"ffmpeg -i {original_video} -i {soundless_video} -c:v copy -c:a copy "
        f"-strict experimental -map 1:v:0 -map 0:a:0 -shortest {final_audio_video}"
    )
    get_logger(__name__).info("Running ffmpeg command: %s", cmd_str)
    os.system(cmd_str)


def subprocess_encode_ffmpeg(
    output_video: str,
    width: int,
    height: int,
    fps: float,
    encoder: str = "h264_nvenc",
    preset: str = "fast",
    gpu_index: int = 0,
):
    # Define the input and output video parameters
    input_video = "input.mp4"
    output_video = "output.mp4"
    width, height = 1920, 1080  # Example width and height

    # OpenCV VideoCapture
    cap = cv2.VideoCapture(input_video)

    # FFmpeg subprocess command with NVENC encoder
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-hide_banner",
        "-f",
        "rawvideo",  # Input format is raw video
        "-vcodec",
        "rawvideo",  # Input codec is raw video
        "-s",
        f"{width}x{height}",  # Size of the input frames
        "-pix_fmt",
        "bgr24",  # OpenCV's default pixel format
        "-r",
        fps,  # Frame rate
        "-i",
        "-",  # Input comes from a pipe
        "-c:v",
        encoder,  # Encoder to use
        "-gpu",
        gpu_index,  # Which GPU to use
        "-preset",
        preset,  # Encoding preset
        output_video,
    ]

    # Open the subprocess
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process your frame here (if needed)
        # ...

        # Write the frame to FFmpeg
        process.stdin.write(frame.tobytes())

    # Cleanup
    cap.release()
    process.stdin.close()
    process.wait()


def get_ffmpeg_decoder_process(
    input_video: str,
    gpu_index: int,
    buffer_size=10**8,
    loglevel: str = "quiet",
    format: str = "bgr24",
    time_s: float = 0.0,
    thread_count: int = 0,
):
    # FFmpeg command for using NVIDIA's hardware decoder
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-hwaccel",
        "cuda",  # Use CUDA hardware acceleration
        "-hwaccel_device",
        str(gpu_index),  # Which GPU to use
        "-ss",
        str(time_s),
        "-i",
        input_video,  # Input file
    ]

    if thread_count:
        command += ["-threads", str(thread_count)]

    command += [
        "-f",
        "image2pipe",  # Output format (pipe)
        "-pix_fmt",
        format,  # Pixel format for OpenCV compatibility
        "-vcodec",
        "rawvideo",  # Output codec (raw video)
        "pipe:1",  # Output to pipe
    ]

    # Start the FFmpeg subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        bufsize=buffer_size,
        preexec_fn=preexec_fn,
    )
    return process


def subprocess_decode_ffmpeg(
    input_video: str,
    # decoder: str = "hevc_cuvid",
    gpu_index: int = 0,
    loglevel: str = "quiet",
):

    # Video parameters
    # width, height = 1920, 1080  # Replace with your video's resolution
    vid_info = BasicVideoInfo(input_video)
    width = vid_info.width
    height = vid_info.height
    channels = 3

    process = get_ffmpeg_decoder_process(
        input_video=input_video, gpu_index=gpu_index, loglevel=loglevel
    )

    # Start the FFmpeg subprocess
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    while True:
        # Read the raw video frame from stdout
        raw_image = process.stdout.read(width * height * channels)

        if not raw_image:
            break

        # Transform the byte read into a numpy array
        frame = np.frombuffer(raw_image, np.uint8).reshape((height, width, channels))

        # Process the frame with OpenCV
        # ...

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Skip to the next frame in the buffer
        process.stdout.flush()

    # Cleanup
    cv2.destroyAllWindows()
    process.stdout.close()
    process.wait()


class VideoWriter:
    def __init__(
        self,
        filename: str,
        fps: float,
        frameSize: Tuple[int],
        isColor: bool = True,
        encoder: str = "hevc_nvenc",
        preset: str = "fast",
        gpu_index: int = 0,
        fake_write: bool = False,
    ):
        self._output_file = filename
        # Ensure a plain float is stored even if callers pass a Fraction
        self._fps = float(fps)
        self._frame_size = frameSize
        self._is_color = isColor
        self._is_openned = False
        self._preset = preset
        self._encoder = encoder
        self._process = None
        self._gpu_index = gpu_index
        self._fake_write = fake_write
        self._open()

    def isOpened(self) -> bool:
        return self._is_openned

    def _open(self):
        assert self._process is None
        # FFmpeg subprocess command with NVENC encoder
        command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-hide_banner",
            "-f",
            "rawvideo",  # Input format is raw video
            "-vcodec",
            "rawvideo",  # Input codec is raw video
            "-s",
            f"{self._frame_size[0]}x{self._frame_size[1]}",  # Size of the input frames
            "-pix_fmt",
            "bgr24",  # OpenCV's default pixel format
            "-r",
            str(self._fps),  # Frame rate
            "-i",
            "-",  # Input comes from a pipe
            "-c:v",
            self._encoder,  # Encoder to use
            "-gpu",
            str(self._gpu_index),  # Which GPU to use
            "-preset",
            self._preset,  # Encoding preset
            self._output_file,
        ]

        # Open the subprocess
        self._process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        self._is_openned = True

    def __del__(self):
        self.release()

    def write(self, frame: np.array):
        if not self._fake_write:
            self._process.stdin.write(frame.tobytes())

    def release(self):
        if self._process is not None:
            self._process.stdout.close()
            self._process.wait()


#!/usr/bin/python
# Filename: ffprobe.py
"""
Python wrapper for ffprobe command line tool. ffprobe must exist in the path.
"""


@classinstancememoize
class FFProbe:
    """
    FFProbe wraps the ffprobe command and pulls the data into an object form::
            metadata=FFProbe('multimedia-file.mov')
    """

    def __init__(self, video_file):
        self.video_file = video_file
        try:
            with open(os.devnull, "w") as tempf:
                subprocess.check_call(["ffprobe", "-h"], stdout=tempf, stderr=tempf)
        except Exception:
            raise IOError("ffprobe not found.")
        if os.path.isfile(video_file):
            if str(platform.system()) == "Windows":
                cmd = ["ffprobe", "-show_streams", self.video_file]
            else:
                cmd = ["ffprobe -show_streams " + self.video_file]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            self.format = None
            self.created = None
            self.duration = None
            self.start = None
            self.bitrate = None
            self.streams = []
            self.video = []
            self.audio = []
            datalines = []
            for a in iter(p.stdout.readline, b""):
                a = a.decode("utf-8")
                if re.match("\[STREAM\]", a):
                    datalines = []
                elif re.match("\[\/STREAM\]", a):
                    self.streams.append(FFStream(datalines))
                    datalines = []
                else:
                    datalines.append(a)
            for a in iter(p.stderr.readline, b""):
                a = a.decode("utf-8")
                if re.match("\[STREAM\]", a):
                    datalines = []
                elif re.match("\[\/STREAM\]", a):
                    self.streams.append(FFStream(datalines))
                    datalines = []
                else:
                    datalines.append(a)
            p.stdout.close()
            p.stderr.close()
            for a in self.streams:
                if a.isAudio():
                    self.audio.append(a)
                if a.isVideo():
                    self.video.append(a)
        else:
            raise IOError("No such media file " + video_file)


class FFStream:
    """
    An object representation of an individual stream in a multimedia file.
    """

    def __init__(self, datalines):
        for a in datalines:
            if "=" in a:
                (key, val) = a.strip().split("=")
            self.__dict__[key] = val

    def isAudio(self):
        """
        Is this stream labelled as an audio stream?
        """
        val = False
        if self.__dict__["codec_type"]:
            if str(self.__dict__["codec_type"]) == "audio":
                val = True
        return val

    def isVideo(self):
        """
        Is the stream labelled as a video stream.
        """
        val = False
        if self.__dict__["codec_type"]:
            if self.codec_type == "video":
                val = True
        return val

    def isSubtitle(self):
        """
        Is the stream labelled as a subtitle stream.
        """
        val = False
        if self.__dict__["codec_type"]:
            if str(self.codec_type) == "subtitle":
                val = True
        return val

    def frameSize(self):
        """
        Returns the pixel frame size as an integer tuple (width,height) if the stream is a video stream.
        Returns None if it is not a video stream.
        """
        size = None
        if self.isVideo():
            if self.__dict__["width"] and self.__dict__["height"]:
                try:
                    size = (int(self.__dict__["width"]), int(self.__dict__["height"]))
                except Exception:
                    print(
                        "None integer size %s:%s"
                        % (str(self.__dict__["width"]), str(+self.__dict__["height"]))
                    )
                    size = (0, 0)
        return size

    def pixelFormat(self):
        """
        Returns a string representing the pixel format of the video stream. e.g. yuv420p.
        Returns none is it is not a video stream.
        """
        f = None
        if self.isVideo():
            if self.__dict__["pix_fmt"]:
                f = self.__dict__["pix_fmt"]
        return f

    def frames(self):
        """
        Returns the length of a video stream in frames. Returns 0 if not a video stream.
        """
        f = 0
        if self.isVideo() or self.isAudio():
            if self.__dict__["nb_frames"]:
                try:
                    f = int(self.__dict__["nb_frames"])
                except Exception:
                    return int(self.durationSeconds() * self.realFrameRate())
        return f

    def durationSeconds(self) -> float:
        """
        Returns the runtime duration of the video stream as a floating point number of seconds.
        Returns 0.0 if not a video stream.
        """
        f = 0.0
        if self.isVideo() or self.isAudio():
            if self.__dict__["duration"]:
                try:
                    f = float(self.__dict__["duration"])
                except Exception:
                    f = duration_to_seconds(self.__dict__["TAG:DURATION"])
        return f

    def language(self):
        """
        Returns language tag of stream. e.g. eng
        """
        lang = None
        if self.__dict__["TAG:language"]:
            lang = self.__dict__["TAG:language"]
        return lang

    def codec(self) -> str:
        """
        Returns a string representation of the stream codec.
        """
        codec_name = None
        if self.__dict__["codec_name"]:
            codec_name = self.__dict__["codec_name"]
        return codec_name

    def codecDescription(self):
        """
        Returns a long representation of the stream codec.
        """
        codec_d = None
        if self.__dict__["codec_long_name"]:
            codec_d = self.__dict__["codec_long_name"]
        return codec_d

    def codecTag(self):
        """
        Returns a short representative tag of the stream codec.
        """
        codec_t = None
        if self.__dict__["codec_tag_string"]:
            codec_t = self.__dict__["codec_tag_string"]
            if codec_t == "[0][0][0][0]":
                codec_t = self.codec().upper()
        return codec_t

    def bitrate(self):
        """
        Returns bitrate as an integer in bps
        """
        b = 0
        if self.__dict__["bit_rate"]:
            try:
                b = int(self.__dict__["bit_rate"])
            except Exception:
                if self.__dict__["bit_rate"] != "N/A":
                    print("None integer bitrate")
        return b

    def realFrameRate(self) -> float:
        """
        Returns average frame rate
        """
        b = 0.0
        if self.__dict__["r_frame_rate"]:
            try:
                rate = self.__dict__["r_frame_rate"]
                if rate:
                    tokens = rate.split("/")
                    token_count = len(tokens)
                    if token_count == 1:
                        return float(tokens[0])
                    elif token_count == 2:
                        b = float(tokens[0]) / float(tokens[1])
                    else:
                        raise AssertionError(
                            f"invalid number of tokens ({token_count}) in r_frame_rate string: {rate}"
                        )
            except Exception:
                print("None integer bitrate")
        return b


def concatenate_videos(video_list: List[str], destination_file: str, force: bool = False) -> bool:
    # Ensure video_list is not empty
    if not video_list:
        raise ValueError("The video list is empty.")

    if not force and os.path.exists(destination_file):
        return

    concat_file, _ = os.path.splitext(destination_file)
    concat_file += ".txt"

    # Create a temporary text file to hold the list of video files
    with open(concat_file, "w") as f:
        for video in video_list:
            # Each line must be in the format: file 'filename'
            f.write(f"file '{os.path.abspath(video)}'\n")

    # Build the ffmpeg command to concatenate the videos
    command = [
        "ffmpeg",
        "-hide_banner",
        "-f",
        "concat",  # Tell ffmpeg that we are using a concatenation file
        "-safe",
        "0",  # Disable safety check to allow absolute paths
        "-i",
        concat_file,  # Input is the list of video files
        "-c",
        "copy",  # Copy both video and audio without re-encoding
        destination_file,  # Output file
    ]

    # Run the command and wait for it to complete
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during concatenation: {e}")
        traceback.print_exc()
        return False
