import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\P19_B_post.mp4"

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

ret, frame = cap.read() #gets the first frame

#Code adapted from ChatGPT

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
print(f'angle: {angle}')

# Calculate the center of rotation
center = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
print(f'center: {center}')

# Rotate the image
rotated_image = rotate_image(image, angle, center)

# Find the new dimensions after rotation
rotated_height, rotated_width = rotated_image.shape[:2]

# Calculate the offset to remove black space (slices must be ints)
x_offset = int((rotated_width - width) / 2)
y_offset = int((rotated_height - height) / 2)

pathvid = os.path.splitext(video_to_process)[0] + '_rotated.jpg' #Add _skeleton to video name
# outvid = cv2.VideoWriter(pathvid,fourcc,fps,(int(width),int(height)))


# # Display the original and rotated images
# cv2.imshow('Original Image', image)
# cv2.imshow('Rotated Image', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### STEP 3: Crop so that the box is always in the center and at the same size

#need original width (width, height)
#need how far right is away from the side
#crop by the difference of these two

center_height_box = (y2-y3)/2 + y3
print(f'center box: {center_height_box}')

rotated_image_2 = rotated_image[0:int(center_height_box), int(x1):int(x2)]

cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#now crop so that other side

#now crop so that the bottom is a certian number of pixels away