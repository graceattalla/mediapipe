"""
@author: Emma Mackey
Based on code available from MediaPipe website and previous code created by Eli Kinney-Lang.

Date created: October 20th, 2021

Intended to process videos through Mediapipe and collect frame-by-frame output.

Note: Handedness assumes rear-facing camera type (added code to flip images, as MediaPipe assumes selfie-camera type)

"""

"""
Ideal formatting of DataFrame: single Index (not MultiIndex)
Frame   Time    Marker     X    Y   Z
"""

#Important question: Should we clip videos for each repetition or just keep note of their frame ranges and process the whole video?


import cv2
import mediapipe
import pandas as pd
from numpy import floor as floor
from numpy import ceil  as ceil
from numpy import size as size
from numpy import shape as shape

def process_video(video_to_process, csv_path, skeleton_path, mintrack = 0.5, mindetect = 0.5):

    #Grab the util solutions for hands
    handsModule = mediapipe.solutions.hands 
    drawingModule = mediapipe.solutions.drawing_utils

    #Set base parameters
    numhands = 2

    #Make an empty dataframe to store solution
    df = []
    df_header_list = []

    with handsModule.Hands(static_image_mode=False,max_num_hands=numhands,min_tracking_confidence=mintrack,min_detection_confidence=mindetect) as hands:
        
        #Get the video capture
        cap = cv2.VideoCapture(video_to_process)
        
        #Deal with writing a video out from CV2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        #Direct skeleton video creation / storage
        outvid = cv2.VideoWriter(skeleton_path,fourcc,fps,(int(width),int(height)))
        
        #Check if the camera opened successfully
        if(cap.isOpened()==False):
            print("ERROR! Problem opening video stream or file")
            
        #Read through the video until it is finished
        while(cap.isOpened()):
            #Capture each frame-by-frame
            ret, frame = cap.read()

            
            #Make sure that the frame is correctly loaded
            if ret==True:

                #Process the image, just like above!
                results = hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) #convert BGR to RGB
                
                #Get the image information
                imageHeight, imageWidth, _ = frame.shape
                
                #This is where the difference with JS Is! They use the following:
                ##New approach
                #
                if results.multi_hand_landmarks!=None and results.multi_handedness!=None:
                    
                    #Now loop through each result in multi_hand_landmarks
                    for count, landmark in enumerate(results.multi_hand_landmarks):
                        
                        #Figure out if it is the right hand or not.
                        handednessIdx = results.multi_handedness[count]
                        #breakdown their classificaitonList into our own temp list
                        tempList = list(handednessIdx._fields.values())
                        #separate out to mimic the JS functions
                        classification = tempList[0][0]
                        if classification.label == "Right":
                            isRightHand = True
                        else:
                            isRightHand = False
                        #register what the landmarks are
                        mylandmark = landmark

                        #Data to store
                        myLabel = classification.label
                        myLabelScore = classification.score

                        #Set up our dataframe information
                        #Headers - Frame, Time, Hand, HandScore, PointName 1 - 21 with tuples
                        #Assign header names.
                        headerNameList = list(handsModule.HandLandmark.__members__.keys())             

                        #Put consistant high-level information into our dataframe
                        d = {
                                    "Frame" : cap.get(cv2.CAP_PROP_POS_FRAMES),
                                    "MsTime": cap.get(cv2.CAP_PROP_POS_MSEC),
                                    "Hand"  : myLabel,
                                    "HandScore" : myLabelScore
                            }
                        
                        

                        #Get each point in the handsModule and then get data from it.
                        for num, point in enumerate(handsModule.HandLandmark):
                            
                            #Figure out which point it is representing for the dataframe

                            try:

                                normalizedLandmark = mylandmark.landmark[point]
                                #Check for rounding errors on x/y, then put them to 1
                                if(normalizedLandmark.x > 1):
                                    normalizedLandmark.x = floor(normalizedLandmark.x)
                                    print("Had to change x - Greater than 1, rounded down now.")
                                elif(normalizedLandmark.y > 1):
                                    normalizedLandmark.y = floor(normalizedLandmark.y)
                                    print("Had to change y - Greater than 1, rounded down now.")
                                elif(normalizedLandmark.x < 0):
                                    normalizedLandmark.x = 0
                                    print("Had to change x - Less than 0, set to 0 now.")
                                elif(normalizedLandmark.y<0):
                                    normalizedLandmark.y = 0
                                    print("Had to change y - Less than 0, set to 0 now.")

                                pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                        #         #for idx, hand_handedness in enumerate(results.multi_handedness): #handedness struggle
                        #             #print(hand_handedness.classification[idx].label) #does this work? (would be defining as variables)

                                d[headerNameList[num]] = [normalizedLandmark.x,normalizedLandmark.y,normalizedLandmark.z]
                                    # "Pixel" + headerNameList[num] : [pixelCoordinatesLandmark[0],pixelCoordinatesLandmark[1],normalizedLandmark.z],
                                    # "Pixel" + headerNameList[num] + "X" : pixelCoordinatesLandmark[0],
                                    # "Pixel" + headerNameList[num] + "Y" : pixelCoordinatesLandmark[1],
                                    # headerNameList[num] + "X" : normalizedLandmark.x,
                                    # headerNameList[num] + "Y" : normalizedLandmark.y,
                                    # headerNameList[num] + "Z" : normalizedLandmark.z
                                


                            ##Draw the Utility parts
                            #Draw on the skeleton
                                if myLabel == "Left":
                                    
                                    drawingModule.draw_landmarks(frame, mylandmark, handsModule.HAND_CONNECTIONS, drawingModule.DrawingSpec(color = (0,0,255), thickness = 2, circle_radius = 2), drawingModule.DrawingSpec(color = (244,244,244)))

                                if myLabel == "Right":

                                    drawingModule.draw_landmarks(frame, mylandmark, handsModule.HAND_CONNECTIONS, drawingModule.DrawingSpec(color = (255,0,0), thickness = 2, circle_radius = 2))

                                if myLabel == "None":
                                    
                                    drawingModule.draw_landmarks(frame, mylandmark, handsModule.HAND_CONNECTIONS, drawingModule.DrawingSpec(color = (0,0,0), thickness = 2, circle_radius = 1))
                                
                            except Exception as e:
                                print(e)
                                print("I don't know what happened. Have fun.")
                                continue

                        #Append our dataframe
                        df.append(d)
                    #
                #

                outvid.write(frame)
                        
                
            else:
                break

            
        #Release the outvid
        outvid.release()
        #Now release everything
        cap.release()
        #Save the output.
        output_df = pd.DataFrame(df)

        #Save to csv file (can open in Excel)
        output_df.to_csv(csv_path)

        return output_df
    

video = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\Test Preprocess 6s\1\P19_B_post_cropped_6s_720p - Copy_preprocessed.mp4"
csv_path=r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\Test Preprocess 6s\1"
skeleton_path=r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Preprocessing\Test Preprocess 6s\1"
process_video(video, csv_path, skeleton_path)