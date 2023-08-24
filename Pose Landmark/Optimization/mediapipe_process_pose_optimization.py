'''
Important Note: This optimization does not work as the Pose model outputs a landmark even for very low presence scores.
-->The minimization function (% frames with a detected landmark) therefore does not work.

Run MediaPipe model at various hyperparameters for multiple videos and save outputs to a csv (parallelization is used when running MediaPipe)

Input: MediaPipe Video

Output: csv of % frames with landmarks detected for various parameters

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

model_path = r'C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Github\mediapipe\Pose_Landmark\pose_landmarker_heavy.task'

#Process folder of videos
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


#Process video with mediapipe hand model
def process_video(video_to_process, mindetect, minpres, mintrack):

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

  # mindetect = 0.8
  # minpres = 0.8
  # mintrack = 0.8

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
    pathvid = os.path.splitext(video_to_process)[0] + '_skeleton.mp4' #Add _skeleton to video name
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
    hands = ["Right", "Left"]
    cor_type = ["Image", "World"]
    information = ["x", "y", "z", "Visibility", "Presence"]

    #setting the column names for the coordinates
    for type in cor_type:
        for r_l in hands:
            for i in range(1, 22):
                for info in information:
                    cor_keys.append(f"{info} {type} {i} {r_l}")

    #setting the column names for the handedness
    handedness_keys = ["Index Right", "Score Right", "Display Name Right", "Category Name Right", 
                        "Index Left", "Score Left", "Display Name Left", "Category Name Left"]

    all_keys.extend(handedness_keys)
    all_keys.extend(cor_keys)  
        
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
        print(pose_landmarker_result)

        # #set dictionary with all frame keys for each frame- reset for each frame
        # d_frame = {key: None for key in all_keys}

        # #loop through total number of hands and all points for given frame
        # #although this is a little redundant, it accounts for issues when there is no data for a certain point and leave it as None
        # for hand, hand_i, hand_w in zip(hand_landmarker_result.handedness, hand_landmarker_result.hand_landmarks, hand_landmarker_result.hand_world_landmarks):
        #   if hand[0].display_name == "Right":
        #     d_frame["Index Right"] = hand[0].index
        #     d_frame["Score Right"] = hand[0].score
        #     d_frame["Display Name Right"] = hand[0].display_name
        #     d_frame["Category Name Right"] = hand[0].category_name

        #     #loop for points in the image and the world coordinates
        #     j = 1
        #     for point_i, point_w in zip(hand_i, hand_w):
        #       j_str = str(j)

        #       d_frame["x Image " + j_str + " Right"] = point_i.x
        #       d_frame["y Image " + j_str + " Right"] = point_i.y
        #       d_frame["z Image " + j_str + " Right"] = point_i.z
        #       d_frame["Visibility Image " + j_str + " Right"] = point_i.visibility
        #       d_frame["Presence Image " + j_str + " Right"] = point_i.presence

        #       d_frame["x World " + j_str + " Right"] = point_w.x
        #       d_frame["y World " + j_str + " Right"] = point_w.y
        #       d_frame["z World " + j_str + " Right"] = point_w.z
        #       d_frame["Visibility World " + j_str + " Right"] = point_w.visibility
        #       d_frame["Presence World " + j_str + " Right"] = point_w.presence

        #       j += 1

        #   if hand[0].display_name == "Left":
        #     d_frame["Index Left"] = hand[0].index
        #     d_frame["Score Left"] = hand[0].score
        #     d_frame["Display Name Left"] = hand[0].display_name
        #     d_frame["Category Name Left"] = hand[0].category_name

        #     #loop for points in the image and the world coordinate
        #     j = 1
        #     for point_i, point_w in zip(hand_i, hand_w):
        #       j_str = str(j)

        #       d_frame["x Image " + j_str + " Left"] = point_i.x
        #       d_frame["y Image " + j_str + " Left"] = point_i.y
        #       d_frame["z Image " + j_str + " Left"] = point_i.z
        #       d_frame["Visibility Image " + j_str + " Left"] = point_i.visibility
        #       d_frame["Presence World " + j_str + " Left"] = point_i.presence

        #       d_frame["x World " + j_str + " Left"] = point_w.x
        #       d_frame["y World " + j_str + " Left"] = point_w.y
        #       d_frame["z World " + j_str + " Left"] = point_w.z
        #       d_frame["Visibility Image " + j_str + " Left"] = point_w.visibility
        #       d_frame["Presence World " + j_str + " Left"] = point_w.presence

        #       j += 1
          
        #   #add the frame dictionary as a new sub-list to the total list
        #   df_list.append(d_frame)

        # #draw landmarks on the image
        annotated_frame = draw_landmarks_on_image(frame, pose_landmarker_result)
        outvid.write(annotated_frame) #write to outvid for each frame        
          
      else:
        break
    output_df = pd.DataFrame(df_list)
    # print(output_df.shape[0])
    # print(output_df.shape[1])

  # #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  # save_path = os.path.splitext(video_to_process)[0] + ".csv"
  # #Save to csv file (can open in Excel)
  # output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return pose_landmarker_result, output_df


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#Skeleton of outputted landmarks
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

#Optimize by brute force for various parameters. Minimization function is percentage of frames with a detected landmark present.
def brute_force_optimization(video_to_process, mindetect, minpres, mintrack):
  #brute force method...
  opt_list = []
  for mindetect in np.arange(0.1, 1.1, 0.1):
    for minpres  in np.arange(0.1, 1.1, 0.1):
      for mintrack  in np.arange(0.5, 1.1, 0.1):
        output_df = process_video(video_to_process, mindetect, minpres, mintrack)[1]
        tot_rows = output_df.shape[0]
        non_none_rows = output_df.iloc[:, 1:].notna().any(axis=1).sum()
        percent_filled = (round(non_none_rows/output_df.shape[0], 2))*100
        print(f"tot row: {output_df.shape[0]}")
        print(f"tot col: {output_df.shape[1]}")
        print(f"detect: {mindetect}")
        print(f"track: {mintrack}")

        row_data = {'Percent Filled': percent_filled, 'Tot Rows': tot_rows, 'Non None Rows': non_none_rows,'Detect': mindetect, 'Presence': minpres, 'Track': mintrack}
        opt_list.append(row_data)

  opt_df = pd.DataFrame(opt_list)


  save_opt_path = os.path.splitext(video_to_process)[0] + "_Optimization" + ".csv"
  opt_df.to_csv(save_opt_path, index=False)

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Hand Legacy\Rotated Videos\Optimization\P19_C_pre_preprocessed_6s.mp4"

brute_force_optimization(video_to_process)