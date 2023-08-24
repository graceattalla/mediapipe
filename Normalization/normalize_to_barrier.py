'''
Normalize coordinates to barrier (collected from barrier_coordiates.py). 
Corrects for translatinal difference in videos. Then corrects for zoom differences in video by making each coordinate a ratio of the barrier height (determined by "barrier_coordinates").

y: Above bottom of barrier is positive
x: to the right of barrier is positive

Input: csv with oroginal coordinates and barrier coordinates
Output: csv with normalized to barrier coordinates

Written by Grace Attalla
'''

import pandas as pd
import os

def process_folder(folder, barrier_file):

    #Go through parent folder and subfolders. Save csv in parent folder (called in function)
    parent_folder = os.listdir(folder) #get a list of the files in the folder

    for inner in parent_folder:

        if os.path.isdir(os.path.join(folder,inner)): #check that it is a folder
            files = os.listdir(os.path.join(folder,inner))

            for file in files:
                
                if ".csv" in file: #only process mp4 videos, can modify if different file type
                    process_path = folder + "\\" + inner +"\\" + file #path to video
                    folder_path = folder + "\\" + inner
                    normalize(process_path, folder_path, barrier_file)


def normalize(file, folder_path, barrier_file):
    cor_df = pd.read_csv(file)
    barrier_df = pd.read_csv(barrier_file)
    file_name = os.path.basename(file)
    file_name = os.path.splitext(file_name)[0]
    print(file_name)

    df_list = []

    #get coordinates of barrier for specified video
    for index, row in barrier_df.iterrows():
        if row["video name"] in file_name:
            y_top = row["y1 normalized"]
            y_bot = row["y2 normalized"]
            x_top = row["x1 normalized"]
            x_bot = row["x2 normalized"]
            barrier_height = y_top-y_bot

    for index, row in cor_df.iterrows():
        d_frame = {}
        #get the normalized coordinates
        for column, value in row.items():
            if "x " in column:
                norm_translation_name = column + " barrier translation norm."
                norm_final_name = column + " barrier total norm"
                if pd.notna(value): #keep as None if nothing there to start with
                    d_frame[norm_translation_name] = value - x_top
                    d_frame[norm_final_name] = (value - x_bot)/barrier_height

                else:
                    d_frame[norm_translation_name] = None
                    d_frame[norm_final_name] = None

            if "y " in column:
                norm_translation_name = column + " barrier translation norm."
                norm_final_name = column + " barrier total norm"
                if pd.notna(value): #keep as None if nothing there to start with
                    d_frame[norm_translation_name] = value - y_bot
                    d_frame[norm_final_name] = (value - y_bot)/barrier_height

                else:
                    d_frame[norm_translation_name] = None
                    d_frame[norm_final_name] = None

        df_list.append(d_frame)

    output_df = pd.DataFrame(df_list)

    save_path = folder_path + "\\" + "barrier_" + file_name + ".csv"
    
    output_df.to_csv(save_path)

folder = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%"
barrier_file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\Barrier_coordinates_estimated_straight_barrier.csv"
save_path = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Barrier Normalization test\test.csv"

process_folder(folder, barrier_file)



