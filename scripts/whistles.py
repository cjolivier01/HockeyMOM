import numpy as np
from scipy.signal import find_peaks

import ffmpeg
import librosa


def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file.
    """
    (
        ffmpeg.input(video_path)
        .output(audio_path, ac=1)  # ac=1 converts audio to mono
        .overwrite_output()
        .run(quiet=True)
    )


def detect_whistles(audio_path, sr=22050):
    """
    Detects whistle sounds in an audio file.
    Returns a list of timestamps where whistles are detected.
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute the Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(y))

    # Convert the amplitude to decibels
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Sum the energy in the high-frequency range typical of whistles
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    high_freq_indices = np.where(frequencies > 2000)[0]  # Frequencies above 2kHz
    high_freq_energy = S_db[high_freq_indices, :].mean(axis=0)

    # Find peaks in the high-frequency energy
    peaks, _ = find_peaks(
        high_freq_energy, height=np.percentile(high_freq_energy, 95), distance=sr * 0.5 / 512
    )

    # Convert frame indices to time
    times = librosa.frames_to_time(peaks, sr=sr)

    return times


def main(video_file):
    audio_file = "temp_audio.wav"

    print("Extracting audio from video...")
    extract_audio(video_file, audio_file)

    print("Detecting whistles...")
    whistle_times = detect_whistles(audio_file)

    print("Whistles detected at the following times (in seconds):")
    for t in whistle_times:
        print(f"{t:.2f}s")

    # Clean up temporary audio file
    import os

    os.remove(audio_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python detect_whistles.py path_to_video.mp4")
    else:
        main(sys.argv[1])
