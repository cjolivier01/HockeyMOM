"""
Experiments in stitching
"""
import os
import time
import cv2
from pathlib import Path
import numpy as np
from lib.opts import opts
from lib.ffmpeg import copy_audio
from lib.ui.mousing import draw_box_with_mouse
from lib.tracking_utils.log import logger

# from lib.tiff import print_geotiff_info
from lib.stitch_synchronize import (
    # synchronize_by_audio,
    # build_stitching_project,
    # extract_frames,
    configure_video_stitching,
)

from lib.datasets.dataset.stitching import (
    StitchDataset,
)


# def configure_video_stitching(
#     dir_name: str,
#     video_left: str = "left.mp4",
#     video_right: str = "right.mp4",
#     project_file_name: str = "my_project.pto",
# )
#     lfo, rfo = synchronize_by_audio(
#         file0_path=os.path.join(dir_name, video_left),
#         file1_path=os.path.join(dir_name, video_right),
#         seconds=15,
#     )

#     base_frame_offset = 800

#     left_image_file, right_image_file = extract_frames(
#         dir_name,
#         video_left,
#         base_frame_offset + lfo,
#         video_right,
#         base_frame_offset + rfo,
#     )

#     # PTO Project File
#     pto_project_file = os.path.join(dir_name, project_file_name)

#     build_stitching_project(
#         pto_project_file, image_files=[left_image_file, right_image_file]
#     )

#     pto_project_file, lfo, rfo = setup_stitching_project(
#         dir_name, video_left, video_right, project_file_name
#     )
#     return lfo, rfo


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    project_file_name: str = "my_project.pto",
):
    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
    )

    start_frame_number = 200
    # start_frame_number = 0

    max_frames = 100

    output_stitched_video_file = "./stitched_output.avi"

    data_loader = StitchDataset(
        video_file_1=f"{dir_name}/{video_left}",
        video_file_2=f"{dir_name}/{video_right}",
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
    )

    frame_count = 0
    start = None
    for i, stitched_image in enumerate(data_loader):
        print(f"Read frame {start_frame_number + i}")
        frame_count += 1
        if i == 1:
            start = time.time()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )
    return lfo, rfo


def pyramid_blending(img1, img2, mask, num_levels=6):
    # Generate Gaussian pyramids for img1, img2 and mask
    gp_img1 = [img1.copy()]
    gp_img2 = [img2.copy()]
    gp_mask = [mask.copy()]
    for i in range(num_levels):
        gp_img1.append(cv2.pyrDown(gp_img1[i]))
        gp_img2.append(cv2.pyrDown(gp_img2[i]))
        gp_mask.append(cv2.pyrDown(gp_mask[i]))

    # Generate Laplacian pyramids for img1 and img2
    lp_img1 = [gp_img1[num_levels - 1]]
    lp_img2 = [gp_img2[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        lp_img1.append(cv2.subtract(gp_img1[i - 1], cv2.pyrUp(gp_img1[i])))
        lp_img2.append(cv2.subtract(gp_img2[i - 1], cv2.pyrUp(gp_img2[i])))

    # Blend images at each level
    blended_pyramid = []
    for l1, l2, m in zip(lp_img1, lp_img2, gp_mask):
        blended_pyramid.append(l1 * m + l2 * (1.0 - m))

    # Reconstruct the image
    blended = blended_pyramid[0]
    for i in range(1, num_levels):
        blended = cv2.pyrUp(blended)
        blended = cv2.add(blended, blended_pyramid[i])

    return blended


def stitch_images(img1_path: str, img2_path: str):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Upper Left  (    0.0,    0.0)
    # Lower Left  (    0.0, 1559.0)
    # Upper Right ( 2595.0,    0.0)
    # Lower Right ( 2595.0, 1559.0)
    # Center      ( 1297.5,  779.5)
    print_geotiff_info("/mnt/data/Videos/vacaville/my_project0000.tif")

    # Upper Left  (    0.0,    0.0)
    # Lower Left  (    0.0, 1628.0)
    # Upper Right ( 2569.0,    0.0)
    # Lower Right ( 2569.0, 1628.0)
    # Center      ( 1284.5,  814.0)
    print_geotiff_info("/mnt/data/Videos/vacaville/my_project0001.tif")

    # Create a mask
    mask = np.zeros_like(img1)
    # Assuming that the left part of img1 and the right part of img2 should be retained, and the overlap should be in the center
    mask[:, : img1.shape[1] // 2, :] = 1

    # Perform pyramid blending
    blended = pyramid_blending(img1, img2, mask)

    # Display the result
    cv2.imshow("Blended Image", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # stitch_images(
    #     "/mnt/data/Videos/vacaville/my_project0000.tif",
    #     "/mnt/data/Videos/vacaville/my_project0001.tif",
    # )

    dir_name = os.path.join(os.environ["HOME"], "Videos", "sabercats-parts")
    video_left = "left-1.mp4"
    video_right = "right-1.mp4"
    # video_left = "left-1-small.avi"
    # video_right = "right-1-small.avi"
    lfo = 0
    rfo = 92
    lfo, rfo = stitch_videos(
        dir_name,
        video_left,
        video_right,
        lfo=lfo,
        rfo=rfo,
    )

    # lfo, rfo = 0, 91
    # if lfo < 0:
    #     copy_audio(
    #         video_left, output_video_path, output_video_with_audio_path
    #     )
    pass


if __name__ == "__main__":
    main()
    print("Done.")
