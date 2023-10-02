import librosa
import numpy as np

# Load the audio file
file_path = '/mnt/home/colivier-local/Videos/output.mp3'
y, sr = librosa.load(file_path, sr=None)

# Compute the spectrogram
S = np.abs(librosa.stft(y))

# Find the frequency and time where the energy of the signal is maximum
freq_idx, time_idx = np.unravel_index(np.argmax(S), S.shape)
frequency = freq_idx * sr / S.shape[0]
time = time_idx * len(y) / sr / S.shape[1]

print(f"A whistle-like sound detected at frequency {frequency:.2f} Hz and time {time:.2f} seconds.")

