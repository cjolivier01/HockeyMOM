import cv2
import os
import uuid
import numpy as np
import subprocess


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


def subprocess_decode_ffmpeg(input_video: str, decoder: str = "h264_cuvid"):
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
