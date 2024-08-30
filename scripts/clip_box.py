import argparse
import os
from pathlib import Path
from typing import List

import cv2

from hmlib.config import (
    get_game_config,
    get_nested_value,
    save_game_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts


def select_clip_box(game_id: str, rect_coords: List[int] = [], final_ar: float = 16 / 9):

    # Load an image
    img = cv2.imread(f'{os.environ["HOME"]}/Videos/{game_id}/s.png')

    if img is None:
        print("Error: Image could not be read.")
        return None

    # Initialize global variables
    drawing = False  # True if mouse is pressed
    ix, iy = -1, -1  # Initial x and y coordinates
    # rect_coords = []  # Stores the coordinates of the rectangle
    scale_width = 0.5  # Scaling factor for the width
    scale_height = 0.5  # Scaling factor for the height

    # Mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img, rect_coords, scale_width, scale_height

        # Translate mouse coordinates to original image coordinates
        x_orig = int(x / scale_width)
        y_orig = int(y / scale_height)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x_orig, y_orig
            rect_coords = [ix, iy, ix, iy]  # Initialize rectangle coordinates

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                rect_coords[2], rect_coords[3] = (
                    x_orig,
                    y_orig,
                )  # Update bottom right corner coords

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect_coords[2], rect_coords[3] = (
                x_orig,
                y_orig,
            )  # Finalize bottom right corner coords

    # Resize image
    height, width = img.shape[:2]
    resized_img = cv2.resize(
        img, (int(width * scale_width), int(height * scale_height))
    )

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        img_copy = resized_img.copy()
        if len(rect_coords) == 4:
            # Draw rectangle on resized image for visualization
            top_left = (
                int(rect_coords[0] * scale_width),
                int(rect_coords[1] * scale_height),
            )
            bottom_right = (
                int(rect_coords[2] * scale_width),
                int(rect_coords[3] * scale_height),
            )
            cv2.rectangle(img_copy, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow("image", img_copy)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            cropped_height = rect_coords[3] - rect_coords[1]
            cropped_width = cropped_height * final_ar
            print(
                f"Rectangle coordinates on original image: {rect_coords}, AR{final_ar}, "
                f"cropped size: {int(cropped_width)} x {int(cropped_height)}"
            )
            break
        elif k == 27:  # Esc key
            rect_coords = None
            break

    cv2.destroyAllWindows()
    return rect_coords


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    args.game_id = "sharks-bb1-2"

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / "..")
    game_config = get_game_config(game_id=args.game_id, root_dir=root_dir)

    rect_coords = get_nested_value(game_config, "game.clip_box")
    if rect_coords is None:
        rect_coords = []

    points = select_clip_box(args.game_id, rect_coords=rect_coords)

    if points and len(points) == 4:
        set_nested_value(game_config, "game.clip_box", points)
        save_game_config(game_id=args.game_id, root_dir=root_dir, data=game_config)
