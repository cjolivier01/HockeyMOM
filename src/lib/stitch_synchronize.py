import os
import moviepy.editor as mp
import librosa
import numpy as np

def synchronize_by_audio(file1_path: str, file2_path: str):
    # Load the videos
    print("Openning videos...")
    video1 = mp.VideoFileClip(file1_path)
    video2 = mp.VideoFileClip(file2_path)

    # Load audio from the videos
    print("Loading audio...")
    audio1 = video1.audio.to_soundarray()
    audio2 = video2.audio.to_soundarray()

    # Calculate the cross-correlation of audio1 and audio2
    print("Calculating cross-correlation...")
    correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode='full')
    lag = np.argmax(correlation) - len(audio1) + 1

    # Calculate the time offset in seconds
    fps = video1.fps
    time_offset = lag / fps

    print(f"Time offset: {time_offset}")

    # Synchronize video2 with video1
    print("Creating subclip...")
    synchronized_video2 = video2.subclip(max(0, -time_offset), video2.duration)

    # Write the synchronized video to a file
    print("Writing synchronized file...")
    synchronized_video2.write_videofile('synchronized_video2.mp4', codec='libx264')

    # Close the videos
    video1.close()
    video2.close()

if __name__ == "__main__":
    video_number = 0
    synchronize_by_audio(
        file1_path=f"{os.environ['HOME']}/Videos/left-{video_number}.mp4",
        file2_path=f"{os.environ['HOME']}/Videos/right-{video_number}.mp4",
    )
