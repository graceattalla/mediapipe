import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import cv2
import os
import pandas as pd
import numpy as np
import scipy as scipy

model_path = r'C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Github\mediapipe\hand_landmarker.task'

# Create a hand landmarker instance with the video mode:
def process_video(video_to_process, mindetect, minpres): #optimizer for just min detect for noe

  #CREATE THE TASK
  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  '''
  min_hand_detection_confidence
  The minimum confidence score for the hand detection to be considered successful in palm detection model.
  
  min_hand_presence_confidence
  The minimum confidence score for the hand presence score in the hand landmark detection model. 
  In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, 
  Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) 
  for subsequent landmark detections.	

  min_tracking_confidence
  The minimum confidence score for the hand tracking to be considered successful. 
  This is the bounding box IoU threshold between hands in the current frame and the last frame. 
  In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. 
  Otherwise, it skips the hand detection.

  '''

  numhands = 1
  #mindetect = 0.3
  # minpres = 0.2
  mintrack = 1

  options = HandLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=r'C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Github\mediapipe\hand_landmarker.task'),
      running_mode=VisionRunningMode.VIDEO, num_hands = numhands,
      min_hand_detection_confidence = mindetect, min_hand_presence_confidence = minpres, min_tracking_confidence = mintrack)
  
  with HandLandmarker.create_from_options(options) as landmarker:
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
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_number)

        #set dictionary with all frame keys for each frame- reset for each frame
        d_frame = {key: None for key in all_keys}

        #loop through total number of hands and all points for given frame
        #although this is a little redundant, it accounts for issues when there is no data for a certain point and leave it as None
        for hand, hand_i, hand_w in zip(hand_landmarker_result.handedness, hand_landmarker_result.hand_landmarks, hand_landmarker_result.hand_world_landmarks):
          if hand[0].display_name == "Right":
            d_frame["Index Right"] = hand[0].index
            d_frame["Score Right"] = hand[0].score
            d_frame["Display Name Right"] = hand[0].display_name
            d_frame["Category Name Right"] = hand[0].category_name

            #loop for points in the image and the world coordinates
            j = 1
            for point_i, point_w in zip(hand_i, hand_w):
              j_str = str(j)

              d_frame["x Image " + j_str + " Right"] = point_i.x
              d_frame["y Image " + j_str + " Right"] = point_i.y
              d_frame["z Image " + j_str + " Right"] = point_i.z
              d_frame["Visibility Image " + j_str + " Right"] = point_i.visibility
              d_frame["Presence Image " + j_str + " Right"] = point_i.presence

              d_frame["x World " + j_str + " Right"] = point_w.x
              d_frame["y World " + j_str + " Right"] = point_w.y
              d_frame["z World " + j_str + " Right"] = point_w.z
              d_frame["Visibility World " + j_str + " Right"] = point_w.visibility
              d_frame["Presence World " + j_str + " Right"] = point_w.presence

              j += 1

          if hand[0].display_name == "Left":
            d_frame["Index Left"] = hand[0].index
            d_frame["Score Left"] = hand[0].score
            d_frame["Display Name Left"] = hand[0].display_name
            d_frame["Category Name Left"] = hand[0].category_name

            #loop for points in the image and the world coordinate
            j = 1
            for point_i, point_w in zip(hand_i, hand_w):
              j_str = str(j)

              d_frame["x Image " + j_str + " Left"] = point_i.x
              d_frame["y Image " + j_str + " Left"] = point_i.y
              d_frame["z Image " + j_str + " Left"] = point_i.z
              d_frame["Visibility Image " + j_str + " Left"] = point_i.visibility
              d_frame["Presence World " + j_str + " Left"] = point_i.presence

              d_frame["x World " + j_str + " Left"] = point_w.x
              d_frame["y World " + j_str + " Left"] = point_w.y
              d_frame["z World " + j_str + " Left"] = point_w.z
              d_frame["Visibility Image " + j_str + " Left"] = point_w.visibility
              d_frame["Presence World " + j_str + " Left"] = point_w.presence

              j += 1
          
          #add the frame dictionary as a new sub-list to the total list
          df_list.append(d_frame)

        #draw landmarks on the image
        #annotated_frame = draw_landmarks_on_image(frame, hand_landmarker_result)
        #outvid.write(annotated_frame) #write to outvid for each frame
                  
          
      else:
        break
    output_df = pd.DataFrame(df_list)
    print(output_df.shape[0])
    print(output_df.shape[1])

  #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  save_path = os.path.splitext(video_to_process)[0] + ".csv"
  #Save to csv file (can open in Excel)
  output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return hand_landmarker_result, output_df


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Fatigue_Vids\Cropped\P19\P19_B_post_cropped_6s_1080p.mp4"


#brute force method...
max_num_rows = 0
max_mindetect = 0
max_minpres = 0

opt_list = []

for mindetect in np.arange(0.1, 0.6, 0.1):
  for minpres  in np.arange(0.1, 0.6, 0.1):
    output_df = process_video(video_to_process, mindetect, minpres)[1]
    num_rows = output_df.shape[0]
    if num_rows > max_num_rows:
      max_num_rows = num_rows
      max_mindetect = mindetect
      max_minpres = minpres
      print(f"NEW MAX: {max_num_rows}")
    print(f"rows: {num_rows}")
    print(f"detect: {mindetect}")
    print(f"pres: {minpres}")

    row_data = {'Num Rows': num_rows, 'Detect': mindetect, 'Presence': minpres}
    opt_list.append(row_data)

opt_df = pd.DataFrame(opt_list)


save_opt_path = os.path.splitext(video_to_process)[0] + "_Optimization" + ".csv"
opt_df.to_csv(save_opt_path, index=False)

print(f"max rows: {max_num_rows}")
print(f"max detect: {max_mindetect}")
print(f"max pres: {max_minpres}")


def draw_landmarks_on_image(rgb_image, detection_result): #taken directly from mediapipe example code
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image