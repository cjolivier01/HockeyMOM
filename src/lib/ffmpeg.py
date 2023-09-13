import os
import uuid


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
