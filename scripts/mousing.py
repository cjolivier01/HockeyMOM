import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image

def draw_box_with_mouse(original_image, destroy_all_windows_after: bool = False):
    # Global variables to store the starting and ending coordinates of the box
    start_x, start_y, end_x, end_y = -1, -1, -1, -1
    drawing = False
    if isinstance(original_image, str):
        original_image = cv2.imread(original_image)
    image = original_image.copy()
    # Mouse callback function
    def draw_box(event, x, y, flags, param):
        nonlocal start_x, start_y, end_x, end_y, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            start_x, start_y = x, y
            end_x, end_y = x, y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_x, end_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            end_x, end_y = x, y
            drawing = False

    cv2.namedWindow('Draw Box')
    cv2.setMouseCallback('Draw Box', draw_box)

    while True:
        # Copy the original image to a temporary image
        temp_image = image.copy()

        # Draw the box on the temporary image
        if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
            cv2.rectangle(temp_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Show the temporary image
        cv2.imshow('Draw Box', temp_image)

        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to clear the drawn box
        if key == ord('c'):
            start_x, start_y, end_x, end_y = -1, -1, -1, -1
            image = original_image.copy()
        # Press 'q' to quit
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    # Return the box
    print(start_x, start_y, end_x, end_y)
    return start_x, start_y, end_x, end_y

def run_selection(image_path: str):
    # Initialize the list to store the bounding boxes
    bounding_boxes = []

    # This callback function will be called when the user draws a rectangle
    def on_draw(event):
        global bounding_boxes
        if isinstance(event.artist, patches.Rectangle):
            bbox = event.artist.get_bbox()
            bounding_boxes.append(bbox)
            print(f"Box drawn: {bbox}")

    # This function will be called when the user clicks the "OK" button
    def on_ok_button_clicked(event):
        print("Final bounding boxes:")
        for bbox in bounding_boxes:
            print(bbox)
        plt.close(fig)

    # Set up the figure and the button
    fig, ax = plt.subplots()
    image = cv2.imread(image_path)
    ax.imshow(image)  # Replace 'your_image' with the variable containing your image
    button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])  # Position of the button
    ok_button = Button(button_ax, 'OK')
    ok_button.on_clicked(on_ok_button_clicked)

    # Connect the callback function to the 'draw_event'
    fig.canvas.mpl_connect('draw_event', on_draw)

    plt.show()


import cv2

# Function to display the image and get ROIs
def get_rois(image_path):
    bounding_boxes = []
    image = cv2.imread(image_path)
    key = 0

    while key != ord('q'):  # Press 'q' to quit the ROI selection process
        bbox = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
        bounding_boxes.append(bbox)

        # Draw the selected ROI on the image (optional)
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with the drawn rectangle
        cv2.imshow("Image", image)

        key = cv2.waitKey(0) & 0xFF

    # Close the image window
    cv2.destroyAllWindows()

    # Print the final bounding boxes
    print("Final bounding boxes:")
    for bbox in bounding_boxes:
        print(bbox)


def draw_line(image_path: str):
    img = Image.open(image_path)
    img = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Variables to store the start and end points
    points = {'start': None, 'end': None}

    def onclick(event):
        # Check if the click is valid (i.e., within the axes)
        if event.inaxes is not None:
            if points['start'] is None:
                # Record the start point
                points['start'] = (event.xdata, event.ydata)
            else:
                # Record the end point and draw the line
                points['end'] = (event.xdata, event.ydata)
                ax.plot([points['start'][0], points['end'][0]], [points['start'][1], points['end'][1]], 'r-')
                fig.canvas.draw()

                # Print the line endpoints
                #print(f"Line drawn from {points['start']} to {points['end']}")
                #print(f"[{int(points['start'][0])}, {int(points['start'][1])}, {int(points['end'][0])}, {int(points['end'][1])}],")
                print(f"      - [{int(points['start'][0])}, {int(points['start'][1])}, {int(points['end'][0])}, {int(points['end'][1])}]")

                # Reset the points
                points['start'], points['end'] = None, None

    # Connect the event handler
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    #draw_box_with_mouse("/mnt/data/Videos/blackhawks/first_tracked_frame.png")
    # Call the function with the path to your image
    #get_rois("/mnt/data/Videos/blackhawks/first_tracked_frame.png")
    #run_selection("/mnt/data/Videos/blackhawks/first_tracked_frame.png")
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    img_file = os.path.join(current_file_path, "..", "s.png")
    draw_line(img_file)




