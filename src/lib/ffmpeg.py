import os
import uuid

FFMPEG_CUDA_FLAGS = "-hwaccel cuda -hwaccel_output_format cuda"


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # output audio from original
    output_audio_path = f"/tmp/output-audio-{uuid.uuid4().hex}.mp3"
    cmd_str = f"ffmpeg {FFMPEG_CUDA_FLAGS} -i {original_video} -q:a 0 -map a {output_audio_path}"
    print(cmd_str)
    os.system(cmd_str)
    # attach audio to new video
    cmd_str = f"ffmpeg {FFMPEG_CUDA_FLAGS} -i {soundless_video} -i {output_audio_path} -map 0:v -map 1:a -c:v copy -shortest {final_audio_video}"
    print(cmd_str)
    os.system(cmd_str)
    # delete temp audio
    if os.path.isfile(output_audio_path):
        os.unlink(output_audio_path)
