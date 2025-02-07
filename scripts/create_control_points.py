#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import torch
import kornia
import kornia.feature as KF
from scipy.signal import correlate


def extract_audio(video_path, sample_rate=44100):
    """
    Load the video using MoviePy and return a mono audio signal (as a 1D numpy array)
    sampled at sample_rate.
    """
    clip = VideoFileClip(video_path)
    # Get audio as numpy array at the desired sample rate.
    audio_array = clip.audio.to_soundarray(fps=sample_rate)
    # If stereo, average channels to get mono.
    if audio_array.ndim == 2 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)
    return audio_array, clip.duration


def find_sync_offset(audio1, audio2, sample_rate=44100):
    """
    Use cross-correlation (full mode) to find the best relative offset between two audio signals.
    Returns the lag (in samples). A positive lag means that audio1 is delayed relative to audio2.
    """
    corr = correlate(audio1, audio2, mode="full")
    # The lags run from -(len(audio2)-1) to len(audio1)-1.
    lag_arr = np.arange(-len(audio2) + 1, len(audio1))
    best_lag = lag_arr[np.argmax(corr)]
    return best_lag


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_frame(video_path, frame_idx):
    """
    Open a video file with OpenCV and return the frame at frame_idx.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")
    return frame


def run_superpoint_lightglue(frame1, frame2):
    """
    Run Kornia’s SuperPoint and LightGlue on two frames to extract matched keypoints.
    Returns a list of control points, where each control point is a tuple:
      (x_img1, y_img1, x_img2, y_img2)
    """
    # Convert OpenCV BGR frames to RGB.
    img1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Convert to torch tensors (shape [C,H,W]) and scale to [0,1].
    img1_tensor = kornia.image_to_tensor(img1_rgb, keepdim=False).float() / 255.0
    img2_tensor = kornia.image_to_tensor(img2_rgb, keepdim=False).float() / 255.0

    # Many SuperPoint implementations expect grayscale input.
    img1_gray = kornia.color.rgb_to_grayscale(img1_tensor)
    img2_gray = kornia.color.rgb_to_grayscale(img2_tensor)

    # Add a batch dimension: [B, 1, H, W].
    img1_batch = img1_gray.unsqueeze(0)
    img2_batch = img2_gray.unsqueeze(0)

    # Load the models. (Ensure your Kornia version provides these.)
    superpoint = KF.SuperPoint().eval()
    lightglue = KF.LightGlue().eval()

    # Disable gradients for inference.
    with torch.no_grad():
        # Run SuperPoint to detect keypoints and compute descriptors.
        preds1 = superpoint(img1_batch)
        preds2 = superpoint(img2_batch)
        # Now run LightGlue to match the keypoints.
        # (The exact API may vary; here we assume that LightGlue accepts the two predictions
        # and returns a dictionary containing keypoints for image0 and image1.)
        matches = lightglue(preds1, preds2)

    # Extract matched keypoints.
    # (In many implementations, matches is a dict with keys 'keypoints0' and 'keypoints1',
    # each an [N,2] tensor of (x,y) coordinates.)
    kp1 = matches["keypoints0"].cpu().numpy()
    kp2 = matches["keypoints1"].cpu().numpy()

    control_points = []
    for pt1, pt2 in zip(kp1, kp2):
        control_points.append((pt1[0], pt1[1], pt2[0], pt2[1]))
    return control_points


def update_pto_file(pto_file, control_points):
    """
    Read the given PTO file, remove any lines beginning with "c" (the control point lines),
    and then insert new control point lines (one per match) before the final "0" line.
    Each new control point line is written in the format:
       c n 0 x_img1 y_img1 1 x_img2 y_img2
    with coordinates formatted to 6 decimal places.
    """
    with open(pto_file, "r") as f:
        lines = f.readlines()

    # Find the index of the line that is simply "0" (end-of-file marker).
    end_index = None
    for i, line in enumerate(lines):
        if line.strip() == "0":
            end_index = i
            break
    if end_index is None:
        end_index = len(lines)

    # Keep all lines before end_index that are not control point lines.
    new_lines = [line for line in lines[:end_index] if not line.startswith("c ")]

    # Append new control point lines.
    for cp in control_points:
        # Here we assume that the two images are image 0 and image 1.
        new_lines.append(f"c n 0 {cp[0]:.6f} {cp[1]:.6f} 1 {cp[2]:.6f} {cp[3]:.6f}\n")

    # Append the remaining lines (including the "0" marker) unchanged.
    new_lines.extend(lines[end_index:])

    # Write the updated content back to the PTO file.
    with open(pto_file, "w") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize two videos using audio cross-correlation, extract sync frames, "
        "compute control points using Kornia SuperPoint+LightGlue, and update a Hugin PTO file."
    )
    parser.add_argument("video1", help="Path to first video file")
    parser.add_argument("video2", help="Path to second video file")
    parser.add_argument("pto_file", help="Path to the Hugin PTO file to update")
    args = parser.parse_args()

    sample_rate = 44100

    print("Extracting audio from videos...")
    audio1, _ = extract_audio(args.video1, sample_rate)
    audio2, _ = extract_audio(args.video2, sample_rate)

    print("Computing cross-correlation to determine sync offset...")
    lag = find_sync_offset(audio1, audio2, sample_rate)
    print(f"Detected lag (in audio samples): {lag}")

    # Get each video’s FPS.
    fps1 = get_video_fps(args.video1)
    fps2 = get_video_fps(args.video2)

    # Interpret the lag:
    # If lag >= 0, then video1’s audio lags behind video2’s; so the sync point is later in video1.
    # Conversely, if lag < 0, video2’s sync point is delayed.
    if lag >= 0:
        sync_time1 = lag / sample_rate
        sync_time2 = 0.0
    else:
        sync_time1 = 0.0
        sync_time2 = -lag / sample_rate

    frame_idx1 = int(round(sync_time1 * fps1))
    frame_idx2 = int(round(sync_time2 * fps2))
    print(
        f"Sync frame indices: video1 -> {frame_idx1} (at {sync_time1:.2f}s), "
        f"video2 -> {frame_idx2} (at {sync_time2:.2f}s)"
    )

    print("Extracting frames at the sync points...")
    frame1 = extract_frame(args.video1, frame_idx1)
    frame2 = extract_frame(args.video2, frame_idx2)

    print("Running SuperPoint and LightGlue to obtain control point matches...")
    control_points = run_superpoint_lightglue(frame1, frame2)
    print(f"Found {len(control_points)} control point matches.")

    print("Updating PTO file with new control points...")
    update_pto_file(args.pto_file, control_points)
    print("PTO file updated successfully.")


if __name__ == "__main__":
    main()
