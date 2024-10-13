import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python script_name.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

try:
    # Load your image
    img = Image.open(image_path)
    data = np.array(img)

    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_title("Click on the image to see pixel values")

    # Event handler to display pixel values
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            pixel_value = data[y, x]
            print(f"Pixel at ({x}, {y}): {pixel_value}")

    # Connect the event to the handler
    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

except FileNotFoundError:
    print(f"Error: The file at {image_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
