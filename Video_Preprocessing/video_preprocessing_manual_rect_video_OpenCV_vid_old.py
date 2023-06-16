import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time

# video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\Full BB Video\P19_B_post_cropped_full_720p.mp4"
video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\P19_B_post_cropped_6s_720p_2.mp4"


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

cv2.imshow('Display Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Code adapted from ChatGPT

### STEP 1: Get the cropping and rotating parameters from the first frame ###
### STEP 1.a: Find the points of the box and blocks rectangle (click in CW direction starting at top left)###
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

start_time = time.time()

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

### STEP 1.b: Get rotation variables ###

# Calculate the angle of rotation
delta_x = top_right[0] - top_left[0]
delta_y = top_right[1] - top_left[1]
angle = np.degrees(np.arctan2(delta_y, delta_x))
print(f'angle: {angle}')

# Calculate the center of rotation
center = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
print(f'center: {center}')

### Step 1.c: Get zoom cropping variables ###

center_height_box = (y2-y3)/2 + y3
print(f'center box: {center_height_box}')

### STEP 2: Apply rotation and zoom to all frames ###

def create_vid(final_image):
    final_height, final_width, channels = final_image.shape
    pathvid = os.path.splitext(video_to_process)[0] + f'_preprocessed.mp4'
    global outvid
    outvid = cv2.VideoWriter(pathvid,fourcc,fps,(int(final_width),int(final_height)))

def rotate_image(image, angle, center):
    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the affine transformation
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated_image

i = 0
while(cap.isOpened()):
    #Capture each frame-by-frame
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #frame number must be an int I think
    print(f"frame: {i}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    ret, frame = cap.read() #frame has image data in numpy array form
    
    #Make sure that the frame is correctly loaded
    if ret==True:

        #apply rotation
        rotated_image = rotate_image(frame, angle, center)

        #apply zooming crop
        final_image = rotated_image[0:int(center_height_box), int(x1):int(x2)]

        if (i == 0):
            create_vid(final_image)

        # cv2.imshow('Final Image', final_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        outvid.write(final_image)
    else:
        break
    i+=1

outvid.release()
cap.release()

end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")