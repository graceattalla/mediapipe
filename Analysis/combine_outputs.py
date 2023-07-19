'''Combine the outputs of two models to try to fill the gaps of missing data for each frame.
Input: CSV files of outputs for videos with 2 different models
Output: New CSV file for each video combining the model outputs. One column will also include which model was used.

Note that hand and holistic label right/left hand oppositely'''

import pandas as pd
import os

def combine_outputs_folder(folder1, folder2): #these should be the parent folders for two different models. Assume folder1 will take preference if data for frame in both.
    pass

def combine_outputs(file1, file2): #two files for same video but from different models

    file1_df = pd.read_csv(file1)
    file2_df = pd.read_csv(file2)

#STEP 1: Create csv with columns including model labels (Each model with a column, fill with 1 or 0)
    df_list = []
    model_keys = ["Hand Right", "Holistic Right", "Hand Left", "Holistic Left"]
    hand_keys = []
    all_keys = []
    hands = ["Right", "Left"]
    cor_type = ["Image", "World"]
    information = ["x", "y", "z"]

    #setting the column names for the coordinates
    for type in cor_type:
        for r_l in hands:
            for i in range(1, 22):
                for info in information:
                    hand_keys.append(f"{info} Imgage {i} {r_l}")
    
    all_keys.extend(model_keys)
    all_keys.extend(hand_keys)
    print(all_keys)

#STEP 2.a: Check if the left hand is full in file1. If so, fill with file 1 and label model column. (Assuming model1 takes priority)

#for row in rows of file1
    #if not none (not includindg the index)
        #if not none in columns with left header
            #fill the new csv with the according coordinates
            #assign the file1 model to be a 1 for the left hand
    print(file1_df)
    for row in file1_df.iloc[1:].iterrows(): #skip the first row because it is the header
        if row.iloc[:, 1:].notna() == None: #NO IOLC
            print("hi")

#STEP 2.b: Check if left hand is full in file2. If so and 1.a false, fill with file2 and label model colummn. If 1.a true, just update model column
#STEP 2.c: (if 2.b false) Leave as empty (not detected for either)
#STEP 3: Repeat Step 2 with right hand

file2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Holistic\0.6 Detect 0.9 Track\P01\B&B_54%_P01_B_pre_preprocessed_0.6d_0.9t_0.34%.csv"
file1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Hand\0.1d 0.5p 1.0t\P01\B&B_20_%P01_B_pre_preprocessed_0.1d_0.5p_1t_19.0%.csv"
combine_outputs(file1, file2)

#next: iloc not working for row (need to check if empty but not include 2 index rows)
