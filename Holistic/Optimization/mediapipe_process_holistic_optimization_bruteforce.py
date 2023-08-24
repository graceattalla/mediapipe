'''
Run MediaPipe model at various hyperparameters for a multiple videos and save outputs to a csv (parallelization is used when running MediaPipe).

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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def draw_landmarks_on_image(image, results):
  image.flags.writeable = True
  # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  annotated_image = np.copy(image)

  #face landmarks

  # mp_drawing.draw_landmarks(
  #     annotated_image,
  #     results.face_landmarks,
  #     mp_holistic.FACEMESH_TESSELATION,
  #     landmark_drawing_spec=None,
  #     connection_drawing_spec=mp_drawing_styles
  #     .get_default_face_mesh_tesselation_style())
  # mp_drawing.draw_landmarks(
  #     annotated_image,
  #     results.face_landmarks,
  #     mp_holistic.FACEMESH_CONTOURS,
  #     landmark_drawing_spec=None,
  #     connection_drawing_spec=mp_drawing_styles
  #     .get_default_face_mesh_contours_style())

  mp_drawing.draw_landmarks(
      annotated_image,
      results.pose_landmarks,
      mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
  mp_drawing.draw_landmarks(
    annotated_image,
    results.left_hand_landmarks,
    mp_holistic.HAND_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles
    .get_default_hand_landmarks_style())
  mp_drawing.draw_landmarks(
    annotated_image,
    results.right_hand_landmarks,
    mp_holistic.HAND_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles
    .get_default_hand_landmarks_style())
  # # Flip the image horizontally for a selfie-view display.
  # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
  # if cv2.waitKey(5) & 0xFF == 27:
  #   break
  return annotated_image

#Process video with holistic model
def process_video(video_to_process, mindetect):

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
  mintrack = 0.9
  numhands = 1


  # options = PoseLandmarkerOptions(
  #     base_options=BaseOptions(model_asset_path=model_path),
  #     running_mode=VisionRunningMode.VIDEO, 
  #     min_pose_detection_confidence = mindetect, min_pose_presence_confidence = minpres, min_tracking_confidence = mintrack)
  
  with mp_holistic.Holistic(
    min_detection_confidence=mindetect,
    min_tracking_confidence=mintrack, model_complexity=2) as holistic: #refine_face_landmarks = True
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
    pathvid = os.path.splitext(video_to_process)[0] + f'_skeleton_{mindetect}d_{mintrack}t.mp4' #Add _skeleton to video name
    outvid = cv2.VideoWriter(pathvid,fourcc,fps,(int(width),int(height)))
    
    #Check if the camera opened successfully
    if(cap.isOpened()==False):
        print("ERROR! Problem opening video stream or file")


    #Get the current position of the video file in milliseconds
    position_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

    #initialize and set column of dataframe
    df_list = []
    cor_hand_keys = []
    cor_pose_keys = []
    all_keys = []
    hands = ["Right", "Left"]
    hand_information = ["x", "y", "z"]
    pose_information = ["x", "y", "z", "visibility"]

    #setting the column names for the hand coordinates
    for r_l in hands:
      for i in range(1, 22):
          for info in hand_information:
              cor_hand_keys.append(f"{i} {info} {r_l} Hand")

    #setting the column names for the pose coordinates
    for i in range (1, 34):
      for info in pose_information:
        cor_pose_keys.append(f"{i} {info} Pose") #1 x Pose
        
    all_keys.extend(cor_hand_keys)  
    all_keys.extend(cor_pose_keys)  
  
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
        holistic_landmarker_result = holistic.process(frame)

        #set dictionary with all frame keys for each frame- reset for each frame
        d_frame = {key: None for key in all_keys}

        #loop through right and left hand of all points for given frame
        #although this is a little redundant, it accounts for issues when there is no data for a certain point and leave it as None
        #can't use zip in for loop becuase hand landmarks and pose landmarks are of different length

        #iterate through points of right landmark if it appears
        j = 1
        if holistic_landmarker_result.right_hand_landmarks != None:
          for point in holistic_landmarker_result.right_hand_landmarks.landmark:
            num_str = str(j)

            d_frame[num_str + " x" + " Right Hand"] = point.x
            d_frame[num_str + " y" + " Right Hand"] = point.y
            d_frame[num_str + " z" + " Right Hand"] = point.z

            j += 1

        #iterate through points of left landmark if it appears
        j = 1
        if holistic_landmarker_result.left_hand_landmarks != None:
          for point in holistic_landmarker_result.left_hand_landmarks.landmark:
            num_str = str(j)

            d_frame[num_str + " x" + " Left Hand"] = point.x
            d_frame[num_str + " y" + " Left Hand"] = point.y
            d_frame[num_str + " z" + " Left Hand"] = point.z

            j += 1

        j = 1


        #commented to generate csv without this info for parameter optimization
        # if holistic_landmarker_result.pose_landmarks != None:
        #   for point in holistic_landmarker_result.pose_landmarks.landmark:
        #     num_str = str(j)

        #     d_frame[num_str + " x" + " Pose"] = point.x
        #     d_frame[num_str + " y" + " Pose"] = point.y
        #     d_frame[num_str + " z" + " Pose"] = point.z
        #     d_frame[num_str + " visibility" + " Pose"] = point.visibility

        #     j += 1
          
        #add the frame dictionary as a new sub-list to the total list
        df_list.append(d_frame)

        # #draw landmarks on the image
        annotated_frame = draw_landmarks_on_image(frame, holistic_landmarker_result)
        outvid.write(annotated_frame) #write to outvid for each frame         
          
      else:
        break
    output_df = pd.DataFrame(df_list)

    #testing outputs
    print(f"tot row: {output_df.shape[0]}")
    print(f"tot col: {output_df.shape[1]}")

    #for labelling of files based on % of frames detected
    non_none_rows = output_df.notna().any(axis=1).sum()
    percent_filled = round(non_none_rows/output_df.shape[0], 2)
    print(f"non-None row: {non_none_rows}")
    print(f"% data rows: {non_none_rows/output_df.shape[0]}")


  #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  save_path = os.path.splitext(video_to_process)[0] + f"_{mindetect}d_{mintrack}t_{percent_filled}%.csv"
  #Save to csv file (can open in Excel)
  output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return holistic_landmarker_result, output_df

#brute force method...

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Holistic\Fatigue Cropped\P05\P05_B_post_6s_1080p.mp4"

max_num_rows = 0
max_mindetect = 0
max_minpres = 0

opt_list = []

#Brute force run though different hyperparameters
for mindetect in np.arange(0.3, 1.0, 0.1):
  output_df = process_video(video_to_process, mindetect)[1]
  non_none_rows = output_df.notna().any(axis=1).sum()
  if non_none_rows > max_num_rows:
    max_num_rows = non_none_rows
    max_mindetect = mindetect
    print(f"NEW MAX: {max_num_rows}")
  print(f"non none rows: {non_none_rows}")
  print(f"detect: {mindetect}")

  row_data = {'Num Rows': non_none_rows, 'Detect': mindetect}
  opt_list.append(row_data)

opt_df = pd.DataFrame(opt_list)


save_opt_path = os.path.splitext(video_to_process)[0] + "_Optimization" + ".csv"
opt_df.to_csv(save_opt_path, index=False)
