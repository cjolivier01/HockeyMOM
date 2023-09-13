import os
import uuid


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # attach audio to new video
    cmd_str = f"ffmpeg -i {output_audio_path} -i {soundless_video} -c:v copy -c:a acc -strict experimental -map 1:v:0 -map 0:a:0 -shortest {final_audio_video}"
    print(cmd_str)
    os.system(cmd_str)
