import os
import moviepy.editor as mp
import numpy as np


def synchronize_by_audio(
    file0_path: str, file1_path: str, seconds: int = 15, create_new_clip: bool = False
):
    # Load the videos
    print("Openning videos...")
    full_video0 = mp.VideoFileClip(file0_path)
    full_video1 = mp.VideoFileClip(file1_path)

    video0 = full_video0.subclip(0, seconds)
    video1 = full_video1.subclip(0, seconds)

    video_1_frame_count = video0.fps * video0.duration
    video_2_frame_count = video1.fps * video0.duration

    # Load audio from the videos
    print("Loading audio...")
    audio1 = video0.audio.to_soundarray()
    audio2 = video1.audio.to_soundarray()

    audio_items_per_frame_1 = audio1.shape[0] / video_1_frame_count
    audio_items_per_frame_2 = audio2.shape[0] / video_2_frame_count

    # Calculate the cross-correlation of audio1 and audio2
    print("Calculating cross-correlation...")
    correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode="full")
    lag = np.argmax(correlation) - len(audio1) + 1

    # Calculate the time offset in seconds
    fps = video0.fps
    frame_offset = lag / audio_items_per_frame_1
    time_offset = frame_offset / fps

    print(f"Left frame offset: {frame_offset}")
    print(f"Time offset: {time_offset} seconds")

    # Synchronize video1 with video0
    if create_new_clip:
        print("Creating new subclip...")
        if frame_offset:
            if frame_offset < 0:
                synchronized_video = full_video1.subclip(
                    max(0, -time_offset), full_video1.duration
                )
                new_file_name = add_suffix_to_filename(file1_path, "sync")
            else:
                synchronized_video = full_video0.subclip(
                    max(0, -time_offset), full_video0.duration
                )
                new_file_name = add_suffix_to_filename(file0_path, "sync")

            # Write the synchronized video to a file
            print("Writing synchronized file...")
            synchronized_video.write_videofile(new_file_name, codec="libx264")
            synchronized_video.close()

    # Close the videos
    video0.close()
    video1.close()
    full_video0.close()
    full_video1.close()

    # Adjust to the starting frame number in each video (i.e. frame_offset might be a negative number)
    left_frame_offset = int(frame_offset if frame_offset > 0 else 0)
    right_frame_offset = int(-frame_offset if frame_offset < 0 else 0)

    return left_frame_offset, right_frame_offset


def add_suffix_to_filename(filename, suffix):
    base_name, extension = os.path.splitext(filename)
    new_filename = f"{base_name}_{suffix}{extension}"
    return new_filename


if __name__ == "__main__":
    # video_number = 0
    # Currently, expects files to be named like
    # "left-0.mp4", "right-0.mp4" and in /home/Videos directory
    synchronize_by_audio(
        file0_path=f"{os.environ['HOME']}/Videos/sabercats-parts/left-1.mp4",
        file1_path=f"{os.environ['HOME']}/Videos/sabercats-parts/right-1.mp4",
        # file0_path=f"{os.environ['HOME']}/Videos/left-{video_number}.mp4",
        # file1_path=f"{os.environ['HOME']}/Videos/right-{video_number}.mp4",
    )
