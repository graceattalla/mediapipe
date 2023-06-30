'''Using parallelization and going through subfolders. Using the mediapipe holistic model'''


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_landmarks_on_image(image, results):
  image.flags.writeable = True
  # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  annotated_image = np.copy(image)
  if results.multi_hand_landmarks != None:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
  # mp_drawing.draw_landmarks(
  #   annotated_image,
  #   results.multi_hand_landmarks,
  #   mp_hands.HAND_CONNECTIONS,
  #   landmark_drawing_spec=mp_drawing_styles
  #   .get_default_hand_landmarks_style())
  # # Flip the image horizontally for a selfie-view display.
  # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
  # if cv2.waitKey(5) & 0xFF == 27:
  #   break
  return annotated_image

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
#take in mindetect and mintrack to optimize
def process_video(video_to_process, mindetect, mintrack):

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

  # mindetect = 0.1
  # mintrack = 0.1
  numhands = 1


  # options = PoseLandmarkerOptions(
  #     base_options=BaseOptions(model_asset_path=model_path),
  #     running_mode=VisionRunningMode.VIDEO, 
  #     min_pose_detection_confidence = mindetect, min_pose_presence_confidence = minpres, min_tracking_confidence = mintrack)
  i = 0

  with mp_hands.Hands(static_image_mode=False,
    min_detection_confidence=mindetect,
    min_tracking_confidence=mintrack, max_num_hands=numhands) as hands: #refine_face_landmarks = True
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
    all_keys = []
    hands_key = ["Right", "Left"]
    hand_information = ["x", "y", "z"]
    cor_type = ["Image", "World"]

        #setting the column names for the coordinates
    for type in cor_type:
        for r_l in hands_key:
            for i in range(1, 22):
                for info in hand_information:
                    cor_hand_keys.append(f"{info} {type} {i} {r_l}")

    #handedness keys
    handedness_keys = ["Index Right", "Score Right", "Label Right", 
                       "Index Left", "Score Left", "Label Left"]
        
    all_keys.extend(handedness_keys)  
    all_keys.extend(cor_hand_keys)  

    # print(all_keys)

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
        # print(hands)
        hand_landmarker_result = hands.process(frame)

        #set dictionary with all frame keys for each frame- reset for each frame
        d_frame = {key: None for key in all_keys}

        #loop through total number of hands and all points for given frame
        #although this is a little redundant, it accounts for issues when there is no data for a certain point and leave it as None
        if hand_landmarker_result.multi_handedness != None:
          for hand, hand_i, hand_w in zip(hand_landmarker_result.multi_handedness, hand_landmarker_result.multi_hand_landmarks, hand_landmarker_result.multi_hand_world_landmarks):
            # print(f"-----{hand_i.landmark}")
            if hand.classification[0].label == "Right":
              d_frame["Label Right"] = hand.classification[0].label
              d_frame["Index Right"] = hand.classification[0].index
              d_frame["Score Right"] = hand.classification[0].score

              #loop for points in the image and the world coordinates
              j = 1
              for point_i, point_w in zip(hand_i.landmark, hand_w.landmark):
                j_str = str(j)

                d_frame["x Image " + j_str + " Right"] = point_i.x
                d_frame["y Image " + j_str + " Right"] = point_i.y
                d_frame["z Image " + j_str + " Right"] = point_i.z

                d_frame["x World " + j_str + " Right"] = point_w.x
                d_frame["y World " + j_str + " Right"] = point_w.y
                d_frame["z World " + j_str + " Right"] = point_w.z

                j += 1

            if hand.classification[0].label == "Left":
              d_frame["Label Left"] = hand.classification[0].label
              d_frame["Index Left"] = hand.classification[0].index
              d_frame["Score Left"] = hand.classification[0].score
              #loop for points in the image and the world coordinate
              j = 1
              for point_i, point_w in zip(hand_i.landmark, hand_w.landmark):
                j_str = str(j)

                d_frame["x Image " + j_str + " Left"] = point_i.x
                d_frame["y Image " + j_str + " Left"] = point_i.y
                d_frame["z Image " + j_str + " Left"] = point_i.z

                d_frame["x World " + j_str + " Left"] = point_w.x
                d_frame["y World " + j_str + " Left"] = point_w.y
                d_frame["z World " + j_str + " Left"] = point_w.z

                j += 1
          
        #add the frame dictionary as a new sub-list to the total list
        df_list.append(d_frame)

        # #draw landmarks on the image
        annotated_frame = draw_landmarks_on_image(frame, hand_landmarker_result)
        outvid.write(annotated_frame) #write to outvid for each frame            
          
      else:
        break
    output_df = pd.DataFrame(df_list)

    #testing outputs
    print(f"tot row: {output_df.shape[0]}")
    print(f"tot col: {output_df.shape[1]}")

    #for labelling of files based on % of frames detected
    non_none_rows = output_df.notna().any(axis=1).sum()
    percent_filled = (round(non_none_rows/output_df.shape[0], 2))*100
    print(f"non-None row: {non_none_rows}")
    print(f"% data rows: {non_none_rows/output_df.shape[0]}")


  #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  save_path = os.path.splitext(video_to_process)[0] + f"_{mindetect}d_{mintrack}t_{percent_filled}%.csv"
  #Save to csv file (can open in Excel)
  output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return hand_landmarker_result, output_df

def brute_force_optimization(video_to_process):
  #brute force method...
  opt_list = []
  for mindetect in np.arange(0.1, 1.1, 0.1):
    for mintrack  in np.arange(0.1, 1.1, 0.1):
      output_df = process_video(video_to_process, mindetect, mintrack)[1]
      tot_rows = output_df.shape[0]
      non_none_rows = output_df.iloc[:, 1:].notna().any(axis=1).sum()
      percent_filled = (round(non_none_rows/output_df.shape[0], 2))*100
      print(f"tot row: {output_df.shape[0]}")
      print(f"tot col: {output_df.shape[1]}")
      print(f"detect: {mindetect}")
      print(f"track: {mintrack}")

      row_data = {'Percent Filled': percent_filled, 'Tot Rows': tot_rows, 'Non None Rows': non_none_rows,'Detect': mindetect, 'Track': mintrack}
      opt_list.append(row_data)

  opt_df = pd.DataFrame(opt_list)


  save_opt_path = os.path.splitext(video_to_process)[0] + "_Optimization" + ".csv"
  opt_df.to_csv(save_opt_path, index=False)

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Hand Legacy\Rotated Videos\Optimization\P19_C_pre_preprocessed_6s.mp4"
brute_force_optimization(video_to_process)

# process_video(r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Hand Legacy\Rotated Videos\Optimization\P19_C_pre_preprocessed_6s")
