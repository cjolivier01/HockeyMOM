import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hmlib.config import (
    get_game_config,
    get_nested_value,
    save_game_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts


def draw_line(image_path: str):
    img = Image.open(image_path)
    img = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Variables to store the start and end points
    points = {"start": None, "end": None}

    def onclick(event):
        # Check if the click is valid (i.e., within the axes)
        if event.inaxes is not None:
            if points["start"] is None:
                # Record the start point
                points["start"] = (event.xdata, event.ydata)
            else:
                # Record the end point and draw the line
                points["end"] = (event.xdata, event.ydata)
                ax.plot(
                    [points["start"][0], points["end"][0]],
                    [points["start"][1], points["end"][1]],
                    "r-",
                )
                fig.canvas.draw()

                # Print the line endpoints
                print(
                    f"      - [{int(points['start'][0])}, {int(points['start'][1])}, {int(points['end'][0])}, {int(points['end'][1])}]"
                )

                # Reset the points
                points["start"], points["end"] = None, None

    # Connect the event handler
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def other_impl(
    game_id: str, image_path: str, upper_lines: List[int] = [], lower_lines: List[int] = []
):
    img = cv2.imread(image_path)

    # Initialize global variables
    is_upper = True
    # upper_lines = []  # To store line endpoints
    # lower_lines = []
    current_line = []  # To store the current line being drawn
    drawing = False  # True if mouse is pressed
    aborted = False

    print("UPPER (type 'U' for upper, 'L' for lower)")

    def draw_line(event, x, y, flags, param):
        nonlocal current_line, drawing, is_upper, img

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_line = [[x, y]]  # Start point of the line

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_line.append([x, y])  # End point of the line
            # Ensure the line is always left to right
            start, end = sorted(current_line)
            if is_upper:
                upper_lines.append([start, end])
            else:
                lower_lines.append([start, end])
            cv2.line(img, start, end, (255, 0, 0) if not is_upper else (0, 0, 255), 2)

    def preprocess_lines(lines: List[List[int]], is_upper: bool):
        for l in lines:
            x = l[0]
            y = l[1]
            w = l[2] - l[0]
            h = l[3] - l[1]
            current_line = [[x, y]]  # Start point of the line
            # current_line.append([w, h])
            current_line.append([l[2], l[3]])
            # Ensure the line is always left to right
            start, end = sorted(current_line)
            cv2.line(img, start, end, (255, 0, 0) if not is_upper else (0, 0, 255), 2)

    def display_image_and_draw_lines(image_path):
        nonlocal img, is_upper, aborted, upper_lines, lower_lines
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a window that can be resized
        cv2.setMouseCallback("Image", draw_line)
        # draw_all_lines(upper_lines, is_upper=True)
        # draw_all_lines(lower_lines, is_upper=False)
        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF
            lines = upper_lines if is_upper else lower_lines
            if key == 27:  # ESC key
                if lines:
                    lines.pop()  # Remove the last drawn line
                    img = cv2.imread(image_path)  # Reload the image to clear drawings
                    for line in lines:  # Redraw remaining lines
                        cv2.line(
                            img,
                            line[0],
                            line[1],
                            (255, 0, 0) if not is_upper else (0, 0, 255),
                            2,
                        )
            elif key in [85, 117]:  # U, u
                print("UPPER")
                is_upper = True
            elif key in [76, 108]:  # L, l
                print("LOWER")
                is_upper = False
            elif key == ord("a") or key == ord("A"):
                aborted = True
            elif key == ord("c") or key == ord("C"):
                upper_lines = []
                lower_lines = []
            elif key == ord("q"):  # Quit the program
                break

        cv2.destroyAllWindows()

    def to_tlbr(lines):
        new_lines = []
        for line in lines:
            ll = [line[0][0], line[0][1], line[1][0], line[1][1]]
            new_lines.append(ll)
        return new_lines

    display_image_and_draw_lines(image_path)
    if not aborted:
        upper_lines = to_tlbr(upper_lines)
        lower_lines = to_tlbr(lower_lines)

        print(f"Upper: {upper_lines}")
        print(f"Lower: {lower_lines}")

        this_path = Path(os.path.dirname(__file__))
        root_dir = os.path.realpath(this_path / "..")
        game_config = get_game_config(game_id=game_id, root_dir=root_dir)
        if game_config is None:
            game_config = dict()

        set_nested_value(game_config, "game.boundaries.upper", upper_lines)
        set_nested_value(game_config, "game.boundaries.lower", lower_lines)
        save_game_config(game_id=game_id, root_dir=root_dir, data=game_config)
        return True
    return False


def maybe_configure_borders(game_id: str, root_dir: str):
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if get_nested_value(game_config, "game.boundaries.upper", None) is None and get_nested_value(
        game_config, "game.boundaries.lower", None
    ):
        # TODO: need to extract frame, why is panorama.tif not always correct?
        other_impl(game_id=game_id, image_path=image_path)


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    assert args.game_id

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / "..")
    game_config = get_game_config(game_id=args.game_id, root_dir=root_dir)
    if game_config:
        upper_lines = get_nested_value(game_config, "game.boundaries.upper")
        if True or upper_lines is None:
            upper_lines = []
        lower_lines = get_nested_value(game_config, "game.boundaries.lower")
        if True or lower_lines is None:
            lower_lines = []
    else:
        upper_lines = []
        lower_lines = []

    image_path = f'{os.environ["HOME"]}/Videos/{args.game_id}/s.png'
    other_impl(
        game_id=args.game_id,
        image_path=image_path,
        upper_lines=upper_lines,
        lower_lines=lower_lines,
    )
