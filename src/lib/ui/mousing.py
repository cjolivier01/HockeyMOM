import cv2
import numpy as np

def draw_box_with_mouse(original_image, destroy_all_windows_after: bool = False):
    # Global variables to store the starting and ending coordinates of the box
    start_x, start_y, end_x, end_y = -1, -1, -1, -1
    drawing = False
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
