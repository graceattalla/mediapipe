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

model_path = r'C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Github\mediapipe\Pose_Landmark\pose_landmarker_heavy.task'

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
  
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5, model_complexity=2, refine_face_landmarks = True) as holistic:
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
        holistic_landmarker_result = holistic.process(frame)
        print(holistic_landmarker_result)


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
        annotated_frame = draw_landmarks_on_image(frame, holistic_landmarker_result)
        outvid.write(annotated_frame) #write to outvid for each frame
                  
          
      else:
        break
    # output_df = pd.DataFrame(df_list)
    # print(output_df.shape[0])
    # print(output_df.shape[1])

  # #Get path to save file (splits at the "." to remove the ".mp4" then adds ".csv")
  # save_path = os.path.splitext(video_to_process)[0] + ".csv"
  # #Save to csv file (can open in Excel)
  # output_df.to_csv(save_path)

  outvid.release()
  cap.release()

  return holistic_landmarker_result


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(image, results):
  image.flags.writeable = True
  # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  annotated_image = np.copy(image)

  mp_drawing.draw_landmarks(
      annotated_image,
      results.face_landmarks,
      mp_holistic.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_tesselation_style())
  mp_drawing.draw_landmarks(
      annotated_image,
      results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_contours_style())
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
