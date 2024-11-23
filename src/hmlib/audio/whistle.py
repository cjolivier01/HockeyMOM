import argparse
from typing import List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect whistles in a hockey game audio file.")
    parser.add_argument(
        "--input-file", type=str, default=None, help="Path to the input audio file."
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to the output file for results.", default=None
    )
    parser.add_argument(
        "--whistle_freq_min", type=float, help="Minimum frequency of whistle (Hz).", default=2000.0
    )
    parser.add_argument(
        "--whistle_freq_max", type=float, help="Maximum frequency of whistle (Hz).", default=4000.0
    )
    parser.add_argument(
        "--duration_threshold",
        type=float,
        help="Minimum duration of whistle (seconds).",
        default=0.05,
    )
    parser.add_argument(
        "--energy_threshold_percentile",
        type=float,
        help="Percentile for energy threshold.",
        default=95.0,
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the energy and detected whistles."
    )
    return parser.parse_args()


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return audio_data, sample_rate


def detect_whistles(
    audio_data: np.ndarray,
    sample_rate: int,
    whistle_freq_min: float,
    whistle_freq_max: float,
    duration_threshold: float,
    energy_threshold_percentile: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray, float]:
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio_data)
    stft_db = librosa.amplitude_to_db(np.abs(stft))

    # Compute the frequency axis
    frequencies = librosa.fft_frequencies(sr=sample_rate)

    # Identify the frequency range that corresponds to the whistle
    whistle_freq_indices = np.where(
        (frequencies >= whistle_freq_min) & (frequencies <= whistle_freq_max)
    )[0]

    # Detect energy in the whistle frequency range
    whistle_energy = np.mean(stft_db[whistle_freq_indices, :], axis=0)

    # Threshold the energy to detect whistle occurrences
    whistle_threshold = np.percentile(whistle_energy, energy_threshold_percentile)
    whistle_frames = np.where(whistle_energy > whistle_threshold)[0]
    whistle_times = librosa.frames_to_time(whistle_frames, sr=sample_rate)

    # Filter out occurrences that are too short to be real whistles
    frame_duration = librosa.frames_to_time(1, sr=sample_rate)
    whistle_intervals = get_whistle_intervals(whistle_times, duration_threshold, frame_duration)

    # Return intervals, energy, times, and threshold for plotting
    times = librosa.frames_to_time(np.arange(len(whistle_energy)), sr=sample_rate)
    return whistle_intervals, whistle_energy, times, whistle_threshold


def get_whistle_intervals(
    whistle_times: np.ndarray, duration_threshold: float, frame_duration: float
) -> List[Tuple[float, float]]:
    whistle_intervals: List[Tuple[float, float]] = []
    if len(whistle_times) == 0:
        return whistle_intervals
    start_time: float = whistle_times[0]
    end_time: float = whistle_times[0]
    for time in whistle_times[1:]:
        if time - end_time <= frame_duration:
            # Continuation of the interval
            end_time = time
        else:
            # New interval
            if end_time - start_time >= duration_threshold:
                whistle_intervals.append((start_time, end_time))
            start_time = time
            end_time = time
    # Add the last interval
    if end_time - start_time >= duration_threshold:
        whistle_intervals.append((start_time, end_time))
    return whistle_intervals


def plot_whistle_detection(
    whistle_energy: np.ndarray,
    times: np.ndarray,
    whistle_threshold: float,
    whistle_intervals: List[Tuple[float, float]],
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(times, whistle_energy, label="Whistle Energy")
    plt.axhline(y=whistle_threshold, color="r", linestyle="--", label="Threshold")
    plt.title("Whistle Detection in Hockey Game")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (dB)")
    plt.legend()
    for start, end in whistle_intervals:
        plt.axvspan(start, end, color="green", alpha=0.3)
    plt.show()


def main() -> None:
    args = parse_arguments()

    args.input_file = "/home/colivier/FastVideo/test3/GX010084.aac"
    args.plot = True

    audio_data, sample_rate = load_audio(args.input_file)

    whistle_intervals, whistle_energy, times, whistle_threshold = detect_whistles(
        audio_data=audio_data,
        sample_rate=sample_rate,
        whistle_freq_min=args.whistle_freq_min,
        whistle_freq_max=args.whistle_freq_max,
        duration_threshold=args.duration_threshold,
        energy_threshold_percentile=args.energy_threshold_percentile,
    )

    # Output results
    if args.output_file:
        with open(args.output_file, "w") as f:
            for start, end in whistle_intervals:
                f.write(f"{start:.2f},{end:.2f}\n")
    else:
        print(f"Detected {len(whistle_intervals)} Whistle Occurrences (start_time, end_time):")
        for start, end in whistle_intervals:
            print(f"({start:.2f}, {end:.2f})")

    # Plot if requested
    if args.plot:
        plot_whistle_detection(whistle_energy, times, whistle_threshold, whistle_intervals)


if __name__ == "__main__":
    main()
