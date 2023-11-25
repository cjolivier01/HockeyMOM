import cv2
import os
import uuid

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
