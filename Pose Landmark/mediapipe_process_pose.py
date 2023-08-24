'''
Process a folder of videos with MediaPipe Pose model. Save the outputs to a csv and skeleton video.

MediaPipe Holistic Model: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

Parallelization is used to increase efficiency.

Input: Folder with participant subfolders of videos.
Outputs: For each video, csv of MediaPipe outputs and skeleton video.

Note: Model will output coordinates for all points, even if it was a very low "Presence" or "Visibility".

Written by Grace Attalla
'''

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import os
import pandas as pd
import numpy as np
from  joblib import Parallel, delayed

model_path = r'C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Github\mediapipe\Pose Landmark\pose_landmarker_heavy.task'

def process_folder(folder):
  parent_folder = os.listdir(folder)
  vid_files = []

  for inner in parent_folder:
      inner_folder = os.path.join(folder, inner) #get full path to inner folder

      if os.path.isdir(inner_folder): #check if it is a folder
          inner_files = os.listdir(inner_folder)
          full_file_paths = [os.path.join(inner_folder, file) for file in inner_files if ".mp4" in file]
          vid_files.extend(full_file_paths)
  print(f"vid_files: {vid_files}") #test

  #Run with parallization
  Parallel(n_jobs=-1, verbose=10)(delayed(process_video)(os.path.join(folder, file)) for file in vid_files)


# Create a hand landmarker instance with the video mode:
def process_video(video_to_process):

  #CREATE THE TASK
  BaseOptions = mp.tasks.BaseOptions
  PoseLandmarker = mp.tasks.vision.PoseLandmarker
  PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode
  '''
  min_hand_detection_confidence
  The minimum confidence score for the hand detection to be considered successful in palm detection model.
  
  min_hand_presence_confidence
  The minimum confidence score for the hand presence score in the hand landmark detection model. 
  In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, 
  Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) for subsequent landmark detections.	

  min_tracking_confidence
  The minimum confidence score for the hand tracking to be considered successful. 
  This is the bounding box IoU threshold between hands in the current frame and the last frame. 
  In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. 
  Otherwise, it skips the hand detection.

  '''

  mindetect = 0.8
  minpres = 0.8
  mintrack = 0.8


  options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=VisionRunningMode.VIDEO, 
      min_pose_detection_confidence = mindetect, min_pose_presence_confidence = minpres, min_tracking_confidence = mintrack)
  
  with PoseLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    df_list = []
    df_frame_list = []
    cap = cv2.VideoCapture(video_to_process)
    
    #Deal with writing a video out from CV2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #testing
    print(f"fps: {fps}")
    print(f"Time (s): {total_frames/fps}")

    #Modified to save at same path for skeleton video (easier to keep organized)
    pathvid = os.path.splitext(video_to_process)[0] + f'_{mindetect}d_{minpres}p_{mintrack}t_skeleton.mp4' #Add _skeleton to video name
    outvid = cv2.VideoWriter(pathvid,fourcc,fps,(int(width),int(height)))
    
    #Check if the camera opened successfully
    if(cap.isOpened()==False):
        print("ERROR! Problem opening video stream or file")


    #Get the current position of the video file in milliseconds
    position_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

    #initialize and set column of dataframe
    df_list = []
    cor_keys = []
    all_keys = []
    cor_type = ["Normalized", "World"]
    information = ["x", "y", "z", "Visibility", "Presence"]

    #setting the column names for the coordinates
    for type in cor_type:
          for i in range(0, 33):
              for info in information:
                  cor_keys.append(f"{info} {type} {i}")

    all_keys.extend(cor_keys)  

    i = 1
    #Read through the video until it is finished
    while(cap.isOpened()):
      #Capture each frame-by-frame
      frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #frame number must be an int I think

      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
      timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

      ret, frame = cap.read() #frame has image data in numpy array form

        
        #Make sure that the frame is correctly loaded
      if ret==True:
        #create an array with all the mediapipe images
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        #run the task
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_number)


        #set dictionary with all frame keys for each frame- reset for each frame
        d_frame = {key: None for key in all_keys}

        #loop through total number of hands and all points for given frame
        #although this is a little redundant, it accounts for issues when there is no data for a certain point and leave it as None
        #landmark 15-22 are hand and wrist landmarks (0-32 landmarks total)
        j = 0
        for pose_n, pose_w in zip(pose_landmarker_result.pose_landmarks, pose_landmarker_result.pose_world_landmarks):
          for point_n, point_w in zip(pose_n, pose_w):
            if i == 1:
              print(pose_n)
              i = 0
            # print(j)
            j_str = str(j)
            if j in range(15,23): #only save the hand related landmarks
              d_frame["x Normalized " + j_str] = point_n.x
              d_frame["y Normalized " + j_str] = point_n.y
              d_frame["z Normalized " + j_str] = point_n.z
              d_frame["Visibility Normalized " + j_str] = point_n.visibility
              d_frame["Presence Normalized " + j_str] = point_n.presence

              d_frame["x World " + j_str] = point_w.x
              d_frame["y World " + j_str] = point_w.y
              d_frame["z World " + j_str] = point_w.z
              d_frame["Visibility World " + j_str] = point_w.visibility
              d_frame["Presence World " + j_str] = point_w.presence

            j += 1
          
          #add the frame dictionary as a new sub-list to the total list
          df_list.append(d_frame)

        # #draw landmarks on the image
        annotated_frame = draw_landmarks_on_image(frame, pose_landmarker_result)
        outvid.write(annotated_frame) #write to outvid for each frame
                  
          
      else:
        break
    output_df = pd.DataFrame(df_list)
    print(output_df)
    print(output_df.shape[0])
    print(output_df.shape[1])

    #for labelling of files based on % of frames detected
    non_none_rows = output_df.notna().any(axis=1).sum()
    percent_filled = (round(non_none_rows/output_df.shape[0], 2))*100

  #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  save_path = os.path.splitext(video_to_process)[0] + f"_{mindetect}d_{minpres}p_{mintrack}t_{percent_filled}%.csv"
  #Save to csv file (can open in Excel)
  output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return pose_landmarker_result


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result): #taken directly from mediapipe example code for pose landmark
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image