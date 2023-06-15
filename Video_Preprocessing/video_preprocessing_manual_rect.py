import cv2
import numpy as np
from matplotlib import pyplot as plt

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Fatigue_Vids_Original\P19\P19_B_post.mp4"

cap = cv2.VideoCapture(video_to_process)
    
#Deal with writing a video out from CV2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# while(cap.isOpened()):
#Capture each frame-by-frame
frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #frame number must be an int I think

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

ret, frame = cap.read()

#Code from ChatGPT

### STEP 1: Find the points of the box and blocks rectangle ###
#click in CW direction starting at top left

import cv2
import numpy as np

# Global variables to store the rectangle's coordinates
points = []
rectangle_selected = False

# Mouse callback function
def select_rectangle(event, x, y, flags, param):
    global points, rectangle_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        rectangle_selected = False

        if len(points) == 4:
            rectangle_selected = True

# Load the image
image = np.copy(frame)

# Create a window and bind the mouse callback function to it
cv2.namedWindow('Select Rectangle')
cv2.setMouseCallback('Select Rectangle', select_rectangle)

while True:
    # Clone the original image
    display_image = image.copy()

    # Draw the rectangle defined by the four points
    if rectangle_selected:
        cv2.drawContours(display_image, [np.array(points)], 0, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Select Rectangle', display_image)

    # Exit the loop if 'ESC' key is pressed or rectangle is selected
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or (rectangle_selected and len(points) == 4):
        break

# Check if the rectangle is selected
if rectangle_selected and len(points) == 4:
    # Extract the corner coordinates
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    #assigning for readability in rotation
    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]

    # Print the corner coordinates
    print("Top Left: ({}, {})".format(x1, y1))
    print("Top Right: ({}, {})".format(x2, y2))
    print("Bottom Right: ({}, {})".format(x3, y3))
    print("Bottom Left: ({}, {})".format(x4, y4))

# Close all windows
cv2.destroyAllWindows()

# # Global variables to store the rectangle's coordinates
# top_left = None
# top_right = None
# bottom_left = None
# bottom_right = None
# rectangle_selected = False

# # Mouse callback function
# def select_rectangle(event, x, y, flags, param):
#     global top_left, top_right, bottom_left, bottom_right, rectangle_selected

#     if event == cv2.EVENT_LBUTTONDOWN:
#         top_left = (x, y)
#         rectangle_selected = False

#     elif event == cv2.EVENT_LBUTTONUP:
#         bottom_right = (x, y)
#         rectangle_selected = True

#         # Calculate the remaining corners
#         top_right = (bottom_right[0], top_left[1])
#         bottom_left = (top_left[0], bottom_right[1])

# # Load the image
# image = np.copy(frame)

# # Create a window and bind the mouse callback function to it
# cv2.namedWindow('Select Rectangle')
# cv2.setMouseCallback('Select Rectangle', select_rectangle)

# while True:
#     # Clone the original image
#     display_image = image.copy()

#     # Draw the currently selected rectangle
#     if top_left and bottom_right:
#         cv2.rectangle(display_image, top_left, bottom_right, (0, 255, 0), 2)

#     # Display the image
#     cv2.imshow('Select Rectangle', display_image)

#     # Exit the loop if 'ESC' key is pressed or rectangle is selected
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27 or rectangle_selected:
#         break

# # Check if the rectangle is selected
# if rectangle_selected:
#     # Extract the corner coordinates
#     x1, y1 = top_left
#     x2, y2 = top_right
#     x3, y3 = bottom_left
#     x4, y4 = bottom_right

#     # Print the corner coordinates
#     print("Top Left: ({}, {})".format(x1, y1))
#     print("Top Right: ({}, {})".format(x2, y2))
#     print("Bottom Left: ({}, {})".format(x3, y3))
#     print("Bottom Right: ({}, {})".format(x4, y4))

# # Close all windows
# cv2.destroyAllWindows()

### STEP 2: Rotate so that box is straight ###

def rotate_image(image, angle, center):
    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the affine transformation
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# Calculate the angle of rotation
delta_x = top_right[0] - top_left[0]
delta_y = top_right[1] - top_left[1]
angle = np.degrees(np.arctan2(delta_y, delta_x))

# Calculate the center of rotation
center = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)

# Rotate the image
rotated_image = rotate_image(image, angle, center)

# Find the new dimensions after rotation
rotated_height, rotated_width = rotated_image.shape[:2]

# Calculate the offset to remove black space (slices must be ints)
x_offset = int((rotated_width - width) / 2)
y_offset = int((rotated_height - height) / 2)

# Crop the image to remove black space
cropped_image = rotated_image[y_offset:y_offset + int(height), x_offset:x_offset + int(width)]

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()