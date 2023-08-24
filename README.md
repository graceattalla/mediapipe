# mediapipe
 
This repository is for the processing of videos using MediaPipe models. Video preprocessing, MediaPipe processing (save csv and skeleton), hyperparameter optimization, model comparison and model output combination is included. Summaries of code in each folder are summarized below.

Analysis
    Compare Models
        --> Plots and stats to compare different MediaPipe model outputs
    Fatigue Analysis
        --> Initial attempt at fatigue analysis through amplitude pre/post BCI comparison and frames per oscillation pre/post BCI comparison. None were significant and was not pursued further in timeframe of project. Only videos with high MediaPipe outputs were used.

Box and Blocks Extraction
    --> Save copy of csv files with only Box and Blocks frames (Box and Blocks start/stop timestamps for each video was collected by hand)

Combine Models
    --> Combine outputs of two models and save into another csv. Visualize new combined csv for number of videos above percentage thresholds of frames with detected landmarks.

Hand Landmark
    --> Process folder of videos with MediaPipe Hand Landmarker model (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)  
    Optimization
        --> Various methods of optimization to optimize hyperparameters fora  given video.

Hand Landmark Legacy
    --> Process folder of videos with MediaPipe Hand Legacy model (https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md)  
    Optimization
        --> Optimize multiple videos with brute force method

Holistic
    --> Process folder of videos with MediaPipe Holistic model (https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md#resources)  
    Optimization
        --> Optimize multiple videos with brute force method

Model Maker
    --> Test setting up Model Maker (https://developers.google.com/mediapipe/solutions/model_maker) for transfer learning

Normalization
    --> Determine coordinates of barrier in Box and Blocks video. Normalize MediaPipe output coordinates to the barrier in order to compare data across videos which were all recorded at different zooms and translational positions.

Pose Landmark
    --> Process folder of videos with MediaPipe Holistic model (https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)  
    Optimization
        --> Optimize multiple videos with brute force method

Video Preprocessing
    --> Rotate videos so that the Box and Blocks container is straight. Manual selection of box is required to collect rotation videos, then videos are rotated.
    Old
        --> Old methods of attempting to select Box and Blocks container and rotate video accordingly so video is straight. These methods were not used in preprocessed videos for MediaPipe processing.
    Parallelization Test
        --> Simple test to ensure parallelization is working correctly

Setup & General
    --> MediaPipe setup and experimental setup recommendations.

If there are any further questions, contact Grace Attalla.
