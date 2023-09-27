"""
Experiments in stitching
"""
import os
import torch
import torch.nn as nn
import numpy as np
import cv2

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import tifffile

from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse

import time

from hockeymom import core

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

        self.duration = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.duration = 0.0


def get_vid_metadata(vid):
    vid_cap = cv2.VideoCapture(vid.as_posix())
    return vid_cap.get(7), vid_cap.get(5), vid_cap.get(7) / vid_cap.get(5)


def create_wide_df(df):
    df["event"] = df["event"].shift(-1)
    df["event_attributes"] = df["event_attributes"].shift(-1)
    df["start_time"] = df["time"]
    df["event_time"] = df["time"].shift(-1)
    df["end_time"] = df["time"].shift(-2)
    df = df.query('event not in ["start", "end"]')
    df = df.drop(["time"], axis=1)
    return df


def create_new_interval_vid_df(vid, df, interval=2):
    _, _, length = get_vid_metadata(vid)

    start_times = np.arange(0, length, interval)
    end_times = np.arange(interval - 0.04, length + interval - 0.04, interval)

    if end_times[-1] > length:
        end_times[-1] = length

    vid_id = vid.name[:-4]
    vid_df = pd.DataFrame(
        {
            "vid_id": vid_id,
            "start_time": start_times,
            "end_time": end_times,
            "event": "background",
        }
    )

    old_vid_df = df.query("video_id == @vid_id")

    for event_time, event in zip(old_vid_df.event_time, old_vid_df.event):
        vid_df.loc[np.argmax(event_time < start_times), "event"] = event

    return vid_df


def create_interval_df(vids, df, interval):
    vid_dfs = []
    for vid in vids:
        vid_dfs.append(create_new_interval_vid_df(vid, df, interval))
    return pd.concat(vid_dfs, axis=0).reset_index(drop=True)


def get_frames_interval(vid_path, start_time, end_time, frame_transform=None):
    vid_cap = cv2.VideoCapture(vid_path.as_posix())
    fps = vid_cap.get(5)
    frames = vid_cap.get(7)
    length = vid_cap.get(7) / vid_cap.get(5)

    start_time = round(start_time / 0.04) * 0.04
    end_time = round(end_time / 0.04) * 0.04

    start_frame = timestamp_to_frame[start_time]
    end_frame = timestamp_to_frame[end_time]

    frames = []

    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = vid_cap.read()
    frame = torch.tensor(frame)

    if frame_transform != None:
        frame = frame_transform(frame)

    frames.append(frame)

    i = start_frame + 1

    while i <= end_frame:
        ret, frame = vid_cap.read()
        frame = torch.tensor(frame)

        if frame_transform != None:
            frame = frame_transform(frame)

        frames.append(frame)
        i += 1

    return torch.stack(frames, 0)


def stitch_images(
    left_file: str, right_file: str, video_number: int, callback_fn: callable = None
):
    scale_down_images = True
    image_scale_down_value = 2
    show_image = True
    skip_frame_count = 0
    stop_at_frame_count = 0
    filename_stitched = None
    filename_with_audio = None

    vidcap_left = cv2.VideoCapture(left_file)
    vidcap_right = cv2.VideoCapture(right_file)

    fps = vidcap_left.get(cv2.CAP_PROP_FPS)
    frame_width = int(vidcap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames_left = int(vidcap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_right = int(vidcap_right.get(cv2.CAP_PROP_FRAME_COUNT))

    # assert total_frames_left == total_frames_right
    total_frames = min(total_frames_left, total_frames_right)

    print(f"Video FPS={fps}")
    print(f"Frame count={total_frames}")
    print(f"Input size: {frame_width} x {frame_height}")

    final_frame_width = frame_width * 2
    final_frame_height = frame_height

    if scale_down_images:
        final_frame_width = (frame_width * 2) // int(image_scale_down_value)
        final_frame_height = frame_height // int(image_scale_down_value)
    print(f"Stitched output size: {final_frame_width} x {final_frame_height}")

    if callback_fn is None:
        filename_stitched = f"stitched-output-{video_number}.mov"
        out = cv2.VideoWriter(
            filename=filename_stitched,
            fourcc=cv2.VideoWriter_fourcc(*"XVID"),
            # fourcc=cv2.VideoWriter_fourcc(*"HEVC"),
            fps=fps,
            frameSize=(final_frame_width, final_frame_height),
            isColor=True,
        )
        assert out.isOpened()
        out.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)
        filename_with_audio = f"stitched-output-{video_number}-with-audio.mov"
    else:
        out = None

    r1, frame1 = vidcap_left.read()
    r2, frame2 = vidcap_right.read()

    frame_id = 0
    timer = Timer()
    while r1 and r2:
        # cv2.imwrite("frames/frame%d.png" % count, image)     # save frame as JPEG file
        timer.tic()
        r1, frame1 = vidcap_left.read()
        r2, frame2 = vidcap_right.read()

        if frame_id < skip_frame_count:
            frame_id += 1
            continue

        if stop_at_frame_count and frame_id >= stop_at_frame_count:
            break

        if final_frame_height != frame_height:
            frame1 = cv2.resize(frame1, (final_frame_width // 2, final_frame_height))
            frame2 = cv2.resize(frame2, (final_frame_width // 2, final_frame_height))

        if True:
            combined_frame = stitch_matcher(frame1, frame2)
            print(f"combined_frame.shape={combined_frame.shape}")
        else:
            # Concatenate the frames side-by-side
            combined_frame = cv2.hconcat([frame1, frame2])

        if show_image:
            cv2.imshow("Combined Image", combined_frame)
            cv2.waitKey(0)

        if out is not None:
            out.write(combined_frame)

        if callback_fn is not None:
            if not callback_fn(combined_frame, (final_frame_width, final_frame_height)):
                break

        timer.toc()
        if frame_id % 20 == 0:
            print(
                "Stitching frame {}/{} ({:.2f} fps)".format(
                    frame_id, total_frames, 1.0 / max(1e-5, timer.average_time)
                )
            )
        frame_id += 1
        if frame_id >= total_frames:
            break

    if out is not None:
        out.release()
    vidcap_left.release()
    vidcap_right.release()

    if show_image:
        cv2.destroyAllWindows()

    if filename_with_audio:
        copy_audio(left_file, filename_stitched, filename_with_audio)


def eval(video_number: int, callback_fn: callable = None):
    input_dir = os.path.join(os.environ["HOME"], "Videos")
    # left_file = os.path.join(input_dir, f"left-{video_number}.mp4")
    # right_file = os.path.join(input_dir, f"right-{video_number}.mp4")
    left_file = os.path.join(input_dir, f"left.mp4")
    right_file = os.path.join(input_dir, f"right.mp4")
    return stitch_images(
        left_file=left_file,
        right_file=right_file,
        video_number=video_number,
        callback_fn=callback_fn,
    )


def expand_image_width(input_image, new_width: int):
    # Get the dimensions of the input image
    height, width, channels = input_image.shape

    # Create a new image with twice the width
    output_image = np.zeros((height, int(new_width), channels), dtype=np.uint8)

    # Calculate the padding on both sides
    # padding_left = (new_width - width) // 2
    # padding_right = new_width - width - padding_left

    # Copy the input image to the center of the new image
    # output_image[:, padding_left:padding_left + width, :] = input_image

    output_image[:, 0:width, :] = input_image
    return output_image


def make_transformed_image(input_image, matrix):
    # Calculate the dimensions of the output image
    height, width = input_image.shape[:2]
    transformed_corners = np.array([
        np.dot(matrix, [0, 0, 1]),
        np.dot(matrix, [width, 0, 1]),
        np.dot(matrix, [0, height, 1]),
        np.dot(matrix, [width, height, 1])
    ])
    min_x = int(np.min(transformed_corners[:, 0]))
    max_x = int(np.max(transformed_corners[:, 0]))
    min_y = int(np.min(transformed_corners[:, 1]))
    max_y = int(np.max(transformed_corners[:, 1]))
    output_width = max_x - min_x
    output_height = max_y - min_y

    # Create a new output image with the calculated dimensions
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Calculate the translation matrix to account for the min_x and min_y values
    translation_matrix = np.array([[1, 0, -min_x],
                                    [0, 1, -min_y],
                                    [0, 0, 1]])

    # Combine the translation matrix and the original transformation matrix
    combined_matrix = np.dot(translation_matrix, matrix)

    # Apply the perspective transformation and place it on the output image
    output_image = cv2.warpPerspective(input_image, combined_matrix, (output_width, output_height))
    output_image = cv2.resize(output_image, dsize=[3000, 3000])
    return output_image


def stitch_with_warp():
    # Load the two video files
    video1 = cv2.VideoCapture(f"{vid_dir}/left.mp4")
    video2 = cv2.VideoCapture(f"{vid_dir}/right.mp4")

    video1.set(cv2.CAP_PROP_POS_FRAMES, 217)
    video2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize the video writer to save the stitched video
    output_video = cv2.VideoWriter(
        "stitched_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # Frames per second
        (1920, 1080),
    )  # Width and height of the output video

    # Iterate through the frames of the first video
    frame_id = 0
    while True:
        ret1, image1 = video1.read()
        if not ret1:
            break

        # Read the corresponding frame from the second video
        ret2, image2 = video2.read()
        if not ret2:
            break

        # left_start = int(image1.shape[1] - 500)
        overlap_size = 750
        # image1 = image1[:,-overlap_size:,:]
        # image2 = image2[:,:overlap_size,:]

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        #
        # Left mask
        #
        # Define the coordinates of the ROI (top-left and bottom-right corners)
        # in (x, y) coordinates (even though image shape is [height, width])
        roi_start = (image1.shape[1] - overlap_size, 0)  # Example coordinates
        roi_end = (image1.shape[1] - 1, image1.shape[0] - 1)  # Example coordinates

        # Create a black mask with the same dimensions as the input image
        left_mask = np.zeros_like(gray1)
        #left_mask = np.ones_like(gray1)

        # Fill the ROI region with white
        cv2.rectangle(left_mask, roi_start, roi_end, (255), thickness=cv2.FILLED)

        # cv2.imshow('Left Mask', left_mask)
        # cv2.waitKey(0)

        #
        # Right mask
        #
        # Define the coordinates of the ROI (top-left and bottom-right corners)
        # in (x, y) coordinates (even though image shape is [height, width])
        roi_start = (0, 0)  # Example coordinates
        roi_end = (overlap_size, image1.shape[0] - 1)  # Example coordinates

        # Create a black mask with the same dimensions as the input image
        right_mask = np.zeros_like(gray2)
        #right_mask = np.ones_like(gray2)

        # Fill the ROI region with white
        cv2.rectangle(right_mask, roi_start, roi_end, (255), thickness=cv2.FILLED)

        # cv2.imshow('Right Mask', right_mask)
        # cv2.waitKey(0)

        # Create SIFT detector object
        # sift = cv2.SIFT_create(edgeThreshold=10, contrastThreshold=0.04)
        sift = cv2.SIFT_create()

        # Detect key points and compute descriptors
        # keypoints1, descriptors1 = sift.detectAndCompute(image1, mask=left_mask)
        # keypoints2, descriptors2 = sift.detectAndCompute(image2, mask=right_mask)
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, mask=left_mask)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, mask=right_mask)

        # Perform feature detection and matching (e.g., using SIFT or ORB)
        # Here, we'll use simple feature detection for demonstration purposes
        # detector = cv2.ORB_create()
        # kp1, des1 = detector.detectAndCompute(frame1, None)
        # kp2, des2 = detector.detectAndCompute(frame2, None)

        # Match features and find a homography
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to find good matches
        good_matches = []

        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good_matches.append(m)
            # if m.distance < 0.75 * n.distance:
            #     good_matches.append(m)

        # Sort matches by their distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        # good_matches = good_matches[:10]

        N = 6  # Number of keypoints to consider

        # Draw the first 10 matches
        match_img = cv2.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            good_matches[:N],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow('Match Image', match_img)
        cv2.waitKey(0)

        if len(good_matches) >= 4:
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            matches_mask = np.zeros(len(good_matches), dtype=np.uint8)
            matches_mask[:N] = 1  # Set the first N matches as inliers
            # Compute the Homography matrix
            homography_matrix, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0, mask=matches_mask
            )

            # Use the Homography matrix to warp the images
            # warped_image = cv2.warpPerspective(image2, M, (image1.shape[1] + image2.shape[1], image2.shape[0]))

            #panoramic_image = expand_image_width(image1, image1.shape[1] * 2)
            #cv2.imshow("Big image", panoramic_image)

            # corrected_perspective = cv2.warpPerspective(panoramic_image, homography_matrix, (panoramic_image.shape[1], panoramic_image.shape[0]))
            # corrected_perspective = cv2.warpPerspective(image2, M, (image2.shape[1] * 2, image2.shape[0] * 2))
            # corrected_perspective = cv2.warpPerspective(
            #     image2, homography_matrix, (panoramic_image.shape[1], panoramic_image.shape[0])
            # )

            corrected_perspective = make_transformed_image(image1, homography_matrix)

            # Copy the second image onto the warped image
            # warped_image[0:image2.shape[0], 0:image2.shape[1]] = image2

            # cv2.imshow('Warped Image', warped_image)``
            # cv2.waitKey(0)

            # warped_image = cv2.warpPerspective(image2, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))
            # warped_image = cv2.warpPerspective(image2, M, (int(image1.shape[1] + image2.shape[1]), int(image1.shape[0] * 2)))

            # # Copy the second image onto the warped image
            # #warped_image[0:image2.shape[0], 0:image2.shape[1]] = image2
            cv2.imshow("Warped Image", corrected_perspective)
            # cv2.imshow('Warped Image', panoramic_image)
            cv2.waitKey(0)

            frame_id += 1

    # Release video objects and close the output video writer
    video1.release()
    video2.release()
    output_video.release()
    cv2.destroyAllWindows()


    # if (!TIFFGetField(tiff, TIFFTAG_XPOSITION, &tiff_xpos)) tiff_xpos = -1;
    # if (!TIFFGetField(tiff, TIFFTAG_YPOSITION, &tiff_ypos)) tiff_ypos = -1;
    # if (!TIFFGetField(tiff, TIFFTAG_XRESOLUTION, &tiff_xres)) tiff_xres = -1;
    # if (!TIFFGetField(tiff, TIFFTAG_YRESOLUTION, &tiff_yres)) tiff_yres = -1;
    # if (tiff_xpos != -1 && tiff_xres > 0) xpos = (int)(tiff_xpos * tiff_xres + 0.5);
    # if (tiff_ypos != -1 && tiff_yres > 0) ypos = (int)(tiff_ypos * tiff_yres + 0.5);

def get_tiff_tag_value(tiff_tag):
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator)/denominator


def get_image_geo_position(tiff_image_file: str):
    xpos, ypos = 0, 0
    with tifffile.TiffFile(tiff_image_file) as tif:
        tags = tif.pages[0].tags
        # Access the TIFFTAG_XPOSITION
        x_position = get_tiff_tag_value(tags.get('XPosition'))
        y_position = get_tiff_tag_value(tags.get('YPosition'))
        x_resolution = get_tiff_tag_value(tags.get('XResolution'))
        y_resolution = get_tiff_tag_value(tags.get('YResolution'))
        xpos = int(x_position * x_resolution + 0.5)
        ypos = int(y_position * y_resolution + 0.5)
        print(f"x={xpos}, y={ypos}")
    return xpos, ypos


def pyramid_blending():
    vid_dir = os.path.join(os.environ["HOME"], "Videos")
    orig_files_left = [
        f"{vid_dir}/images/left.png",
        f"{vid_dir}/images/left-45min.png",
    ]

    orig_files_right = [
        f"{vid_dir}/images/right.png",
        f"{vid_dir}/images/right-45min.png",
    ]

    nona = core.HmNona(f"{vid_dir}/my_project.pto")


    for i in range(1):
        for i in range(len(orig_files_left)):
            img1 = cv2.imread(orig_files_left[i % len(orig_files_left)])
            img2 = cv2.imread(orig_files_right[i % len(orig_files_left)])
            # cv2.imshow('Nona image left', img1)
            # cv2.waitKey(0)
            # cv2.imshow('Nona image right', img2)
            # cv2.waitKey(0)
            start = time.time()
            if True:
                result = core.nona_process_images(nona, img1, img2)
                duration = time.time() - start
                print(f"Got results in {duration} seconds")
                cv2.imshow('Nona image left', result[0])
                cv2.waitKey(0)
                cv2.imshow('Nona image right', result[1])
                cv2.waitKey(0)
            else:
                result = core.stitch_images(nona, img1, img2)
                duration = time.time() - start
                print(f"Got results in {duration} seconds")
                # cv2.imshow('Stitched Image', result)
                # cv2.waitKey(0)

    # files_left = [
    #     f"{vid_dir}/my_project0000.tif",
    #     f"{vid_dir}/my_project-20000.tif",
    # ]
    # files_right = [
    #     f"{vid_dir}/my_project0001.tif",
    #     f"{vid_dir}/my_project-20001.tif",
    # ]

    # xpos_1, ypos_1 = get_image_geo_position(files_left[0])
    # xpos_2, ypos_2 = get_image_geo_position(files_right[0])

    # for i in range(len(files_left)):
    #     A = cv2.imread(files_left[i])
    #     B = cv2.imread(files_right[i])

    #     img = core.emblend_images(A, B, [xpos_1, ypos_1], [xpos_2, ypos_2])
    #     # cv2.imshow('Panoramic blended image', img)
    #     # cv2.waitKey(0)

    #cv2.destroyAllWindows()

def main():
    pyramid_blending()
    # core.enblend(f"{vid_dir}/pano-1.png", [
    #     f"{vid_dir}/my_project0000.tif",
    #     f"{vid_dir}/my_project0001.tif",
    # ])
    # eval(video_number=0)
    # stitch_with_warp()
    # panoramic_warp()


if __name__ == "__main__":
    main()
    print("Done.")
