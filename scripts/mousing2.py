import cv2
import numpy as np

# Initialize variables
drawing = False
points = []
segments = []


# Create a callback function for mouse events
def draw_spline(event, x, y, flags, param):
    global drawing, points, image, segments

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            if len(points) > 1:
                # Fit a spline curve to the drawn points
                spline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [spline], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.imshow("Spline Drawing", image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) > 1:
            # Convert the spline into 8 closest-fitting line segments
            spline = np.array(points, dtype=np.float32)
            approximated_points = []
            for t in np.linspace(0, 1, 8):
                x = int(np.interp(t, np.linspace(0, 1, len(points)), spline[:, 0]))
                y = int(np.interp(t, np.linspace(0, 1, len(points)), spline[:, 1]))
                approximated_points.append((x, y))
            new_segments = [
                (approximated_points[i], approximated_points[i + 1]) for i in range(len(approximated_points) - 1)
            ]
            segments.extend(new_segments)
            points = []

            # Draw the closest-fitting line segments
            for segment in new_segments:
                cv2.line(image, segment[0], segment[1], (0, 0, 255), 2)
            cv2.imshow("Spline Drawing", image)


# Create an image canvas
width, height = 800, 600
image = np.zeros((height, width, 3), dtype=np.uint8)

cv2.namedWindow("Spline Drawing")
cv2.setMouseCallback("Spline Drawing", draw_spline)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord("x") or key == ord("X"):
        if len(segments) > 0:
            # Remove the last drawn spline and line segments
            segments.pop()
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for segment in segments:
                cv2.line(image, segment[0], segment[1], (0, 0, 255), 2)
            cv2.imshow("Spline Drawing", image)

cv2.destroyAllWindows()

# Print the list of line segments at the end
print("List of line segments:", segments)
