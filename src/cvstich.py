import os
import cv2
import numpy as np
import math
from hockeymom import core


def _main(pto_project_file):
    input_image = cv2.imread(f"{os.environ['HOME']}/Videos/tvbb2/left.png")

    # Define the field of view (FOV) value in degrees
    fov_degrees = 99.199702905193

    # Camera parameters
    # fx = 1000.0  # Focal length in pixels (along x-axis)
    # fy = 1000.0  # Focal length in pixels (along y-axis)

    image_width = input_image.shape[1]  # Optical center x-coordinate
    image_height = input_image.shape[0]  # Optical center y-coordinate

    # Calculate the focal length (fx and fy) using the FOV formula
    fov_radians = math.radians(fov_degrees)
    focal_length_x = image_width / (2 * math.tan(fov_radians / 2))
    focal_length_y = image_height / (2 * math.tan(fov_radians / 2))

    # Define the camera intrinsic matrix with adjusted focal lengths
    K = np.array(
        [
            [focal_length_x, 0, image_width / 2],
            [0, focal_length_y, image_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Define the radial distortion center shift
    # center_shift_x = 16.167116986623  # Shift in x-coordinate (pixels)
    # center_shift_y = -141.096539158394  # Shift in y-coordinate (pixels)

    center_shift_x = 0.0  # Shift in x-coordinate (pixels)
    center_shift_y = 0.0  # Shift in y-coordinate (pixels)

    # Incorporate the radial distortion center shift into the camera matrix
    K[0, 2] += center_shift_x
    K[1, 2] += center_shift_y

    # Define the distortion coefficients:
    k1 = 0.0  # Radial distortion coefficient
    k2 = 0.0  # Radial distortion coefficient
    k3 = 0.0

    # k1 = 0.0  # Radial distortion coefficient
    # k2 = -0.0490882016017017  # Radial distortion coefficient
    # k2 = 0.0490882016017017  # Radial distortion coefficient
    # k3 = 0.0  # Radial distortion coefficient

    p1 = 0.0  # Tangential distortion coefficient
    p2 = 0.0  # Tangential distortion coefficient

    # p1 = 16.167116986623  # Tangential distortion coefficient
    # p2 = -141.096539158394  # Tangential distortion coefficient

    # roll_deg = 10.0
    # pitch_deg = 5.0
    # yaw_deg = 15.0

    # roll_deg = 0
    # pitch_deg = 12.50508189658689
    # yaw_deg = -42.0477000409216

    distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    roll_deg = 0
    pitch_deg = 0
    yaw_deg = 0

    roll = math.radians(roll_deg)  # Roll angle in radians
    pitch = math.radians(pitch_deg)  # Pitch angle in radians
    yaw = math.radians(yaw_deg)  # Yaw angle in radians

    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ],
        dtype=np.float32,
    )

    Ry = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ],
        dtype=np.float32,
    )

    Rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    R = np.dot(np.dot(Rx, Ry), Rz)

    # new_shape = input_image.shape[1::-1]
    # grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        K,
        distortion_coeffs,
        input_image.shape[1::-1],
        alpha=1,
    )

    # mm = np.dot(new_camera_matrix, R)
    # print(mm)
    # bicubic_warped_image = cv2.warpPerspective(
    #     input_image, mm, input_image.shape[1::-1], flags=cv2.INTER_CUBIC
    # )
    # cv2.imshow("Bicubic Warped Image", bicubic_warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_image = cv2.undistort(
        input_image,
        new_camera_matrix,
        distortion_coeffs,
        # None,
        # new_camera_matrix,
    )

    # output_image = cv2.undistort(
    #     input_image,
    #     K,
    #     distortion_coeffs,
    #     None,
    #     new_camera_matrix,
    # )

    # Warp the image with bicubic interpolation

    # Combine the rotation matrices
    # R = np.dot(np.dot(Rx, Ry), Rz)
    # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, distortion_coeffs, input_image.shape[1::-1], alpha=1)
    # output_image = cv2.warpPerspective(output_image, np.dot(new_camera_matrix, R), input_image.shape[1::-1], flags=cv2.INTER_CUBIC)

    cv2.imshow("Undistorted Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(pto_project_file):
    # Load the input image
    image = cv2.imread(f"{os.environ['HOME']}/Videos/tvbb2/left.png")

    # Camera parameters
    
    k1, k2, k3 = 0.2, 0.1, 0.0
    
    #k1 = 0.0
    #k2 = -0.0490882016017017
    #k3 = 0.0
    
    #k1, k2, k3 = np.deg2rad([k1, k2, k3])
    
    #center_shift = (16.167116986623, -141.096539158394)
    center_shift = (16.167116986623, -141.096539158394)
    #center_shift = (0, 0)
    
    #roll, pitch, yaw = 10.0, 5.0, 15.0  # degrees
    roll, pitch, yaw = 0, 0, 0
    #roll = 0
    #pitch = 12.50508189658689
    #yaw = -42.0477000409216

    fov_horizontal = 100  # Horizontal Field of View in degrees
    fov_vertical = 68    # Vertical Field of View in degrees

    # Convert angles to radians
    roll, pitch, yaw, fov_horizontal, fov_vertical = np.deg2rad([roll, pitch, yaw, fov_horizontal, fov_vertical])

    # Calculate focal length based on FoV
    h, w = image.shape[:2]

    # Calculate focal lengths for horizontal and vertical FoV
    f_x = w / (2 * np.tan(fov_horizontal / 2))
    f_y = h / (2 * np.tan(fov_vertical / 2))

    # Updated camera matrix with center shift
    cx, cy = w / 2 + center_shift[0], h / 2 + center_shift[1]
    camera_matrix = np.array([[f_x, 0, cx],
                            [0, f_y, cy],
                            [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, k3, 0, 0], dtype=np.float32)

    # Apply radial distortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5
    )
    distorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # Rotation matrices for roll, pitch, yaw
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Combined rotation matrix
    R = np.dot(np.dot(R_z, R_y), R_x)

    # Apply rotation
    rotated_image = cv2.warpPerspective(distorted_image, R, (w, h))

    # Display or save the warped image
    cv2.imshow("Linear Warped Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(pto_project_file="/mnt/ripper-data/Videos/tvbb2/autooptimiser_out.pto")
