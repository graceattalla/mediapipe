'''
Save the rotating variables to a csv to access later and preprocess videos. 
Program will ask to click the 4 corners of the box for each video in the specified folder (at bottom of program). 
Start at top left corner and go in CW direction.
'''


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import pandas as pd

def process_folder(folder):

    #Go through parent folder and subfolders. Save csv in parent folder (called in function)
    parent_folder = os.listdir(folder) #get a list of the files in the folder

    for inner in parent_folder:
        print(parent_folder
              )

        if os.path.isdir(os.path.join(folder,inner)): #check that it is a folder
            files = os.listdir(os.path.join(folder,inner))

            for file in files:
                
                if ".mp4" in file: #only process mp4 videos, can modify if different file type
                    process_path = folder + "\\" + inner +"\\" + file #path to video
                    preprocess_variables(process_path)

    save_path = os.path.join(folder, "Preprocessing_variables.csv")
    print(os.path.splitext(folder)[0])
    #Save to csv file (can open in Excel)
    output_df = pd.DataFrame(df_list)
    output_df.to_csv(save_path) 


# video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\Full BB Video\P19_B_post_cropped_full_720p.mp4"
# video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\P19_B_post_cropped_6s_720p_2.mp4"

#setup dataframe to save each video's variables to. Each row contains one video.
columns = ['video path', 'angle', 'center x', 'center y', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
df_list = []


def rotate_image(image, angle, center):
    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the affine transformation
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated_image

#rotate, zoom and save preprocessed video
def preprocess_variables(video_to_process):
    dict_vid = {}
    dict_vid['video path'] = video_to_process

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

    # cv2.imshow('Display Image', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Some code adapted from ChatGPT

    ### STEP 1: Get the cropping and rotating parameters from the first frame ###
    ### STEP 1.a: Find the points of the box and blocks rectangle (click in CW direction starting at top left)###
    # Global variables to store the rectangle's coordinates
    points = []
    rectangle_selected = False

    # Load the image
    image = np.copy(frame)

    # Mouse callback function
    def select_rectangle(event, x, y, flags, param):
        nonlocal points, rectangle_selected #not sure if the nonlocal will cause other problems? I think it can't be global because of outer function?

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            rectangle_selected = False

            if len(points) == 4:
                rectangle_selected = True

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

        dict_vid['x1'] = x1
        dict_vid['y1'] = y1
        dict_vid['x2'] = x2
        dict_vid['y2'] = y2
        dict_vid['x3'] = x3
        dict_vid['y3'] = y3
        dict_vid['x4'] = x4
        dict_vid['y4'] = y4

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
    dict_vid['angle'] = angle
    print(f'angle: {angle}')

    # Calculate the center of rotation
    center = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
    dict_vid['center x'] = center[0]
    dict_vid['center y'] = center[1]
    print(f'center: {center}')

    df_list.append(dict_vid)

    ### Step 1.c: Get zoom cropping variables ###- unnecessary because we will normalize

    # center_height_box = (y2-y3)/2 + y3
    # print(f'center box: {center_height_box}')

    cap.release()

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")
    print(dict_vid)


process_folder(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Preprocess_test\6s test")
# process_folder(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Preprocess_test\6s test")
