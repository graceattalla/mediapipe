'''
Using the variables from video_preprocessing_manual_rect_video_OpenCV_variables_csv.py, rotate each video and save.

The accounts for camera angle rotation differing across each video recorded.

Written by Grace Attalla
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import pandas as pd

def process_folder(folder): #this folder contains the unprocessed videos and a CSV file with their rotation variables.

    start_time = time.time()

    #Get the csv in the parent folder
    files = os.listdir(folder)
    for file in files:
        if ".csv" in file:
            folder_variables = pd.read_csv(folder + "\\" + file)

    parent_folder = os.listdir(folder) #get a list of the files in the folder
    video_num = 1
    for inner in parent_folder:

        if os.path.isdir(os.path.join(folder,inner)): #check that it is a folder
            files = os.listdir(os.path.join(folder,inner))

            for file in files:
                
                if ".mp4" in file: #only process mp4 videos, can modify if different file type
                    print(f"file: {file}")
                    process_path = folder + "\\" + inner +"\\" + file #path to video
                    preprocess_video(process_path, folder_variables)
                    print(f"Video {video_num} complete")
                    video_num += 1

    parent_folder = os.listdir(folder) #get a list of the files in the folder)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution Time:", execution_time, "seconds")

#rotate and save preprocessed video
def create_vid(video_to_process, final_image, fourcc, fps):
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

#rotate, zoom and save preprocessed video
def preprocess_video(video_to_process, folder_variables):
    #retrieving variables from dataframe
    vid_row = folder_variables[folder_variables['video path'] == video_to_process]
    print(f"vid row: {vid_row}")
    angle = vid_row['angle'].item()
    center = (vid_row['center x'].item(), vid_row['center y'].item())
    # print(f"cx: {vid_row['center x']}")
    # print(f"CENTER: {center}")

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

    # Some code adapted from ChatGPT

    ### STEP 2: Apply rotation and zoom to all frames ###

    i = 0
    while(cap.isOpened()):
        #Capture each frame-by-frame
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #frame number must be an int I think
        # print(f"frame: {i}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

        ret, frame = cap.read() #frame has image data in numpy array form
        
        #Make sure that the frame is correctly loaded
        if ret==True:

            #apply rotation
            final_image = rotate_image(frame, angle, center)

            #apply zooming crop- unnecessary because we will normalize
            # final_image = rotated_image[0:int(center_height_box), int(x1):int(x2)]

            if (i == 0):
                create_vid(video_to_process,final_image, fourcc, fps)

            # cv2.imshow('Final Image', final_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            outvid.write(final_image)
        else:
            break
        i+=1

    outvid.release()
    cap.release()

process_folder(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Fatigue_Vids_Original_1min\B Post")