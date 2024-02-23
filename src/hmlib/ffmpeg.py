import cv2
import os
import subprocess
import re
import platform
import numpy as np
import subprocess
from typing import Tuple
import subprocess
import ctypes
import signal
from torchaudio.utils import ffmpeg_utils

from hmlib.utils.utils import classinstancememoize

libc = ctypes.CDLL("libc.so.6")


def preexec_fn():
    # Ensure the child process gets SIGTERM if the parent dies
    libc.prctl(1, signal.SIGTERM)


@classinstancememoize
class BasicVideoInfo:
    def __init__(self, video_file: str, use_ffprobe: bool = True):
        if use_ffprobe:
            probe = FFProbe(video_file)
            self._ffstream: FFStream = None
            if not probe.video:
                raise AssertionError(
                    f"Unable to get video stream from file: {video_file}"
                )
            elif len(probe.video) > 1:
                raise AssertionError(
                    f"Found too many ({len(probe.video)}) video streams in file: {video_file}"
                )
            self._ffstream = probe.video[0]
            self.frame_count = self._ffstream.frames()
            self.fps = self._ffstream.realFrameRate()
            sz = self._ffstream.frameSize()
            self.width = sz[0]
            self.height = sz[1]
            self.bitrate = self._ffstream.bitrate()
            self.codec = self._ffstream.codec()
            self.fourcc = fourcc_to_int(self._ffstream.codec())
        else:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise AssertionError(f"Unable to open video file {video_file}")
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.bitrate = cap.get(cv2.CAP_PROP_BITRATE)
            self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = "".join(
                [chr((self.fourcc >> 8 * i) & 0xFF) for i in range(4)]
            ).upper()
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
    hours, minutes, seconds_ns = duration_str.split(':')
    seconds, ns = seconds_ns.split('.')

    # Convert hours and minutes to seconds, and nanoseconds to seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ns) / 1e9

    return total_seconds


def print_ffmpeg_info():
    print("Library versions:")
    print(ffmpeg_utils.get_versions())
    print("\nBuild config:")
    print(ffmpeg_utils.get_build_config())
    print("\nDecoders:")
    print([k for k in ffmpeg_utils.get_video_decoders().keys() if "cuvid" in k])
    print("\nEncoders:")
    print([k for k in ffmpeg_utils.get_video_encoders().keys() if "nvenc" in k])


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # attach audio to new video
    cmd_str = f"ffmpeg -i {original_video} -i {soundless_video} -c:v copy -c:a copy "
    f"-strict experimental -map 1:v:0 -map 0:a:0 -shortest {final_audio_video}"
    print(cmd_str)
    os.system(cmd_str)


def extract_frame_image(source_video: str, frame_number: int, dest_image: str):
    print(f"Extracting frame {frame_number} from {source_video}...")
    cmd_str = f'ffmpeg -y -i {source_video} -vf "select=eq(n\,{frame_number})" -vframes 1 {dest_image}'
    print(cmd_str)
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
        self._fps = fps
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
        except:
            raise IOError("ffprobe not found.")
        if os.path.isfile(video_file):
            if str(platform.system()) == "Windows":
                cmd = ["ffprobe", "-show_streams", self.video_file]
            else:
                cmd = ["ffprobe -show_streams " + self.video_file]
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )
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
                except Exception as e:
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
                except Exception as e:
                    return int(self.durationSeconds() * self.realFrameRate())
        return f

    def durationSeconds(self):
        """
        Returns the runtime duration of the video stream as a floating point number of seconds.
        Returns 0.0 if not a video stream.
        """
        f = 0.0
        if self.isVideo() or self.isAudio():
            if self.__dict__["duration"]:
                try:
                    f = float(self.__dict__["duration"])
                except Exception as e:
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

    def codec(self):
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
        return codec_t

    def bitrate(self):
        """
        Returns bitrate as an integer in bps
        """
        b = 0
        if self.__dict__["bit_rate"]:
            try:
                b = int(self.__dict__["bit_rate"])
            except Exception as e:
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
            except Exception as e:
                print("None integer bitrate")
        return b
