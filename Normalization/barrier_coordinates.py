'''
Get the barrier image coordinates (top and bottom of barrier) of each video and save to a csv file.

Barrier approximated to be straight if slanted
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

        if os.path.isdir(os.path.join(folder,inner)): #check that it is a folder
            files = os.listdir(os.path.join(folder,inner))

            for file in files:
                
                if ".MP4" in file or ".mp4" in file: #only process mp4 videos, can modify if different file type
                    print(file)
                    process_path = folder + "\\" + inner +"\\" + file #path to video
                    barrier_coordinates(process_path)

    save_path = os.path.join(folder, "Barrier_coordinates.csv")
    #Save to csv file (can open in Excel)
    output_df = pd.DataFrame(df_list)
    output_df.to_csv(save_path) 


# video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\Full BB Video\P19_B_post_cropped_full_720p.mp4"
# video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\P19_B_post_cropped_6s_720p_2.mp4"

#setup dataframe to save each video's variables to. Each row contains one video.
columns = ['video name', 'x1', 'y1', 'x2', 'y2', 'x1 normalized', 'y1 normalized', 'x2 normalized', 'y2 normalized']
df_list = []


#rotate, zoom and save preprocessed video
def barrier_coordinates(video_to_process):
    dict_vid = {}
    name = os.path.basename(video_to_process)
    dict_vid['video name'] = os.path.splitext(name)[0]

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

    # Global variables to store the rectangle's coordinates
    points = []
    done = False

    # Load the image
    image = np.copy(frame)
    y_max = image.shape[0]
    x_max = image.shape[1]

    # Mouse callback function
    def select_barrier(event, x, y, flags, param):
        nonlocal points, done #not sure if the nonlocal will cause other problems? I think it can't be global because of outer function?

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            done = False

            if len(points) == 2:
                done = True

    # Create a window and bind the mouse callback function to it
    cv2.namedWindow('Select Barrier')
    cv2.setMouseCallback('Select Barrier', select_barrier)

    start_time = time.time()

    while True:
        # Clone the original image
        display_image = image.copy()


        # Select top and bottom of barrier
        if done:
            cv2.drawContours(display_image, [np.array(points)], 0, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Select Barrier', display_image)

        # Exit the loop if 'ESC' key is pressed or rectangle is selected
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or (done and len(points) == 2):
            break

    # Check if the barrier is selected
    if done and len(points) == 2:
        # Extract the corner coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]

        #assigning for readability in rotation
        top_left = points[0]
        top_right = points[1]


        dict_vid['x1'] = x1
        dict_vid['y1'] = y1
        dict_vid['x2'] = x2
        dict_vid['y2'] = y2

        dict_vid['x1 normalized'] = x1/x_max
        dict_vid['y1 normalized'] = 1-  y1/y_max #want 1 to be top of the frame
        dict_vid['x2 normalized'] = x2/x_max
        dict_vid['y2 normalized'] = 1 - y2/y_max


        # Print the corner coordinates
        print("Top: ({}, {})".format(x1, y1))
        print("Bottom: ({}, {})".format(x2, y2))
    # Close all windows
    cv2.destroyAllWindows()

    df_list.append(dict_vid)

    ### Step 1.c: Get zoom cropping variables ###- unnecessary because we will normalize

    # center_height_box = (y2-y3)/2 + y3
    # print(f'center box: {center_height_box}')

    cap.release()

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")
    print(dict_vid)

process_folder(r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Normalization test\Folder")
# process_folder(r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%")
# process_folder(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Preprocess_test\6s test")
