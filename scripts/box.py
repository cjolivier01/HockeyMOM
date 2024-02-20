import cv2

# Initialize global variables
points = []  # List to store points

def click_event(event, x, y, flags, params):
    # If left mouse button clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Check if less than 4 points have been clicked
            points.append([x, y])  # Append the point
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Draw circle at the clicked point
            if len(points) > 1:
                cv2.line(img, tuple(points[-2]), tuple(points[-1]), (255, 0, 0), 2)  # Draw line between points
            cv2.imshow('image', img)  # Show the image
        if len(points) == 4:  # If 4 points have been clicked
            cv2.destroyAllWindows()  # Close the window

# Load an image
img = cv2.imread('/home/colivier/Videos/tvbb/panorama.tif')  # Replace 'path_to_your_image.jpg' with your image path
cv2.imshow('image', img)

# Set mouse callback function for window
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()  # Destroy all windows

# Print the four points
if len(points) == 4:
    print(points)

