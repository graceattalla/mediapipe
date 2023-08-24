'''
Using the variables from video_preprocessing_manual_rect_video_OpenCV_variables_csv.py, rotate each video and save.

The accounts for camera angle rotation differing across each video recorded.

Parallelization is used.

Written by Grace Attalla
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import pandas as pd
from joblib import Parallel, delayed

def parallelize_video_processing(folder):
    start_time = time.time()

    #Get the csv in the parent folder
    files = os.listdir(folder)

    for file in files:
        if ".csv" in file:
            folder_variables = pd.read_csv(folder + "\\" + file)

    #store list of full paths of each video
    parent_folder = os.listdir(folder)
    vid_files = []

    for inner in parent_folder:
        inner_folder = os.path.join(folder, inner) #get full path to inner folder

        if os.path.isdir(inner_folder): #check if it is a folder
            inner_files = os.listdir(inner_folder)
            full_file_paths = [os.path.join(inner_folder, file) for file in inner_files if ".mp4" or ".MP4" in file]
            vid_files.extend(full_file_paths)
    # print(f"vid_files: {vid_files}") #test

    #Run with parallization
    Parallel(n_jobs=-1, verbose=10)(delayed(preprocess_video)(os.path.join(folder, file), folder_variables) for file in vid_files)

#For NOT using parallelization
def process_folder(folder): #this folder contains the unprocessed videos and a CSV file with their rotation variables.

    start_time = time.time()

    #Get the csv in the parent folder
    files = os.listdir(folder)
    for file in files:
        if ".csv" in file:
            folder_variables = pd.read_csv(folder + "\\" + file)

    #access each file in subfolders
    parent_folder = os.listdir(folder) #get a list of the files in the folder

    for inner in parent_folder:

        if os.path.isdir(os.path.join(folder,inner)): #check that it is a folder
            files = os.listdir(os.path.join(folder,inner))

            for file in files:
                
                if ".mp4" in file: #only process mp4 videos, can modify if different file type
                    process_path = folder + "\\" + inner +"\\" + file #path to video
                    preprocess_video(process_path, folder_variables)

    end_time = time.time()
    execution_time = end_time - start_time

    # print("Execution Time:", execution_time, "seconds")


#rotate and save preprocessed video
#create path and video to write to
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
    angle = vid_row['angle'].item()
    center = (vid_row['center x'].item(), vid_row['center y'].item())

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
    return i #irrelevant but need to return something?

parallelize_video_processing(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Need to Rotate")
# preprocess_video(r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\P19\MultiTest\P19_B_post_cropped_6s_720p - Copy.mp4")
