import cv2
import os
import uuid
import numpy as np
import subprocess
from typing import Tuple


class BasicVideoInfo:
    def __init__(self, video_file: str):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise AssertionError(f"Unable to open video file {video_file}")
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bitrate = cap.get(cv2.CAP_PROP_BITRATE)
        self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((self.fourcc >> 8 * i) & 0xFF) for i in range(4)]).upper()
        cap.release()


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # attach audio to new video
    cmd_str = f"ffmpeg -i {original_video} -i {soundless_video} -c:v copy -c:a copy -strict experimental -map 1:v:0 -map 0:a:0 -shortest {final_audio_video}"
    print(cmd_str)
    os.system(cmd_str)


def convert_to_h265(source_video: str, dest_video: str):
    # attach audio to new video
    cmd_str = f"/usr/local/bin/ffmpeg -y -hwaccel cuda -i {source_video} -c:v libx265 -crf 40 -b:a 192k -preset medium -tune fastdecode -c:a copy {dest_video}"
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


def subprocess_decode_ffmpeg(
    input_video: str, decoder: str = "h264_cuvid", gpu_index: int = 0
):
    # Video parameters
    width, height = 1920, 1080  # Replace with your video's resolution

    # TODO: get w, h with opencv

    # FFmpeg command for using NVIDIA's hardware decoder
    command = [
        "ffmpeg",
        "-hwaccel",
        "cuda",  # Use CUDA hardware acceleration
        "-hwaccel_output_format",
        "cuda",  # Output format for compatibility
        "-c:v",
        decoder,  # Specify the NVIDIA decoder for H.264
        "-gpu",
        gpu_index,  # Which GPU to use
        "-i",
        input_video,  # Input file
        "-f",
        "image2pipe",  # Output format (pipe)
        "-pix_fmt",
        "bgr24",  # Pixel format for OpenCV compatibility
        "-vcodec",
        "rawvideo",  # Output codec (raw video)
        "pipe:1",  # Output to pipe
    ]

    # Start the FFmpeg subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    while True:
        # Read the raw video frame from stdout
        raw_image = process.stdout.read(width * height * 3)

        if not raw_image:
            break

        # Transform the byte read into a numpy array
        frame = np.frombuffer(raw_image, np.uint8).reshape((height, width, 3))

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


# C version

# // ... Initialization code ...

# // Find the hardware decoder
# const AVCodec* decoder = avcodec_find_decoder_by_name("h264_cuvid");

# // Create a context for the decoder and set any specific options
# AVCodecContext* decoder_ctx = avcodec_alloc_context3(decoder);
# // Set options on decoder_ctx as needed, for example, selecting a GPU

# // Open the decoder
# avcodec_open2(decoder_ctx, decoder, NULL);

# // ... Code to read and decode frames ...

# // Find the hardware encoder
# const AVCodec* encoder = avcodec_find_decoder_by_name("h264_nvenc");

# // Create a context for the encoder and set it up
# AVCodecContext* encoder_ctx = avcodec_alloc_context3(encoder);
# // Set any specific options for the encoder

# // Open the encoder
# avcodec_open2(encoder_ctx, encoder, NULL);

# // ... Code to encode and write frames ...

# // ... Clean-up code ...

# fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
# if not is_cuda:
#     # def __init__(self, filename: str, apiPreference: int, fourcc: int, fps: float, frameSize: cv2.typing.Size, params: _typing.Sequence[int]) -> None: ...
#     # params = Sequence()
#     self._output_video = cv2.VideoWriter(
#         filename=self._output_video_path,
#         # apiPreference=cv2.CAP_FFMPEG,
#         # apiPreference=cv2.CAP_GSTREAMER,
#         fourcc=fourcc,
#         fps=self._fps,
#         frameSize=(
#             int(self._output_frame_width),
#             int(self._output_frame_height),
#         ),
#         # params=[
#         #     cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
#         #     #cv2.VIDEOWRITER_PROP_HW_DEVICE, 1,
#         # ],
#     )


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
