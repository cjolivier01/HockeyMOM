import os
import cv2
from pathlib import Path

from hmlib.config import get_game_config, save_game_config, set_nested_value
from hmlib.hm_opts import hm_opts


def select_opencv(game_id: str):
    # Initialize global variables
    points = []  # List to store points

    def click_event(event, x, y, flags, params):
        # If left mouse button clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:  # Check if less than 4 points have been clicked
                points.append([x, y])  # Append the point
                cv2.circle(
                    img, (x, y), 3, (0, 255, 0), -1
                )  # Draw circle at the clicked point
                if len(points) > 1:
                    cv2.line(
                        img, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2
                    )  # Draw line between points
                cv2.imshow("image", img)  # Show the image

    # Load an image
    img = cv2.imread(
        f'{os.environ["HOME"]}/Videos/{game_id}/s.png'
    )  # Replace 'path_to_your_image.jpg' with your image path
    cv2.imshow("image", img)

    # Set mouse callback function for window
    cv2.setMouseCallback("image", click_event)

    try:
        cv2.waitKey(0)  # Wait indefinitely for a key press
    except Exception as ex:
        print(ex)
    cv2.destroyAllWindows()  # Destroy all windows

    # Print the four points
    if len(points) == 4:
        print(points)
        return points
    return None


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / "..")
    game_config = get_game_config(game_id=args.game_id, root_dir=root_dir)

    points = select_opencv(args.game_id)

    if points and len(points) == 4:
        set_nested_value(game_config, "rink.scoreboard.perspective_polygon", points)
        save_game_config(game_id=args.game_id, root_dir=root_dir, data=game_config)
