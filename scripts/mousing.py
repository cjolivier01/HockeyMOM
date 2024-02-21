import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image


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


if __name__ == "__main__":
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_file_path, "..", "s.png")
    draw_line(img_file)
