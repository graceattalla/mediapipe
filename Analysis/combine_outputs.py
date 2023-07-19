'''Combine the outputs of two models to try to fill the gaps of missing data for each frame.
Input: CSV files of outputs for videos with 2 different models
Output: New CSV file for each video combining the model outputs. One column will also include which model was used.

Note that hand and holistic label right/left hand oppositely'''

import pandas as pd
import os
import re

#To Do:
#- could make it so that 1 and 2 can be interchangable. Just use the folder name for the model columns and take the matching_key function for file1 as well.
    #- maybe don't do this because holistic has swapped right/left so hand to hard code this for file 2.
#- could make filling of csv from file a seperate function
#- NEED TO SWAP WHERE HOLISTIC ACTUALLY SAVES!!!! Save to right if left header name.

def combine_outputs_folder(folder1, folder2): #these should be the parent folders for two different models. Assume folder1 will take preference if data for frame in both.
    pass

def combine_outputs(file1, file2): #two files for same video but from different models. Assumes 1 is hand and 2 is holistic

    file1_df = pd.read_csv(file1, index_col = 0) #set the frame number to be the index column. Use loc to access this
    file2_df = pd.read_csv(file2, index_col = 0)
    file1_df = file1_df.iloc[:, 1:] #extra redundant index column needs to be removed
    file2_df = file2_df.iloc[:, 1:]
    index_list = list(file1_df.index)

#STEP 1: Create csv with columns including model labels (Each model with a column, fill with 1 or 0)
    df_list = []
    model_keys = ["Index", "Hand Right", "Hand Left", "Holistic Right", "Holistic Left"]
    hand_model_info = ["Score Right", "Score Left"]
    hand_keys = []
    all_keys = []
    hands = ["Right", "Left"]
    information = ["x", "y", "z"]

    #setting the column names for the coordinates
    for r_l in hands:
        for i in range(1, 22):
            for info in information:
                hand_keys.append(f"{info} Image {i} {r_l}")
    
    all_keys.extend(model_keys)
    all_keys.extend(hand_model_info)
    all_keys.extend(hand_keys)


#STEP 2.a: Check if the right hand is full in file1. If so, fill with file 1 and label model column. (Assuming model1 takes priority)

#for index in all indicies
    #if index row in file1 has values:
        #for header, value in header, value
            #if this value is left
                #if also has a value
                    #fill new csv with values.
                    #fill left model1 with a 1
                #else
                    #fill with 0
            #if this value is right
                # if also has a value
                    #fill new csv with values.
                    #fill right model1 with a 1
                #else
                    #fill with 0

    #if index row in file2 has a value
        #for header, value in header, value
            #if the value is left
                #if not none:
                    #if new csv not filled yet
                        #fill with values
                        #fill model with 1
                    #else
                        #fill with 0
            #if value is right
                #if not none:
                    #if new csv not filled yet
                    #fill with values
                    #fill model with 1
                #else:
                    #fill with 0
    for master_index in index_list:
    # for master_index, non_row in enumerate(file1_df.iterrows()): #index should be same for file1 and file2
        d_frame = {key: None for key in all_keys} #initialize new row to all None
        d_frame["Index"] = master_index
        row_df1 = file1_df.loc[master_index, :] #need to use loc is that the frame index is used
        row_df2 = file2_df.loc[master_index, :]

        #fill the rows with model1 values if they exist. also update model 1 labels.
        for header, value in zip(file1_df.columns, row_df1):
            if "Right" in header:
                if pd.notna(value): #DON'T use != None in python
                    if header in all_keys: #this is something we want to transfer to the new csv
                        match_key = matching_key(header, all_keys)
                        d_frame[match_key] = value
                        d_frame["Hand Right"] = 1 #this row's right cells are filled with model1
                else:
                    d_frame["Hand Right"] = 0
            if "Left" in header:
                if pd.notna(value):
                    if header in all_keys: #this is something we want to transfer to the new csv
                        match_key = matching_key(header, all_keys)
                        d_frame[match_key] = value
                        d_frame["Hand Left"] = 1 #this row's left cells are filled with model 2
                else:
                    d_frame["Hand Left"] = 0

        #fill remaining empty values with model2 if they exist. also update model 2 labels.
        for header, value in zip(file2_df.columns, row_df2): #header of holistic e.g., 2 x Right Hand
            if "Right" in header:
                if pd.notna(value): #the second model has a value
                    match_key = matching_key(header, all_keys) #not exact matches between each model so need to check
                    d_frame["Holistic Left"] = 1

                    if d_frame[match_key] != None: #has not been filled already by file 1
                        d_frame[match_key] = value
                else: #nothing detected
                    d_frame["Holistic Left"] = 0

            if "Left" in header:
                if pd.notna(value): #the second model has a value
                    match_key = matching_key(header, all_keys) #not exact matches between each model so need to check
                    d_frame["Holistic Right"] = 1

                    if d_frame[match_key] != None: #has not been filled already by file 1
                        d_frame[match_key] = value
                else: #nothing detected
                    d_frame["Holistic Right"] = 0
        df_list.append(d_frame)
        print(master_index)
    output_df = pd.DataFrame(df_list)
    save_path = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Combined Output Test" + "\combined_output.csv"
    output_df.to_csv(save_path)


                    #if corresponding value in dictionary is not filled yet. dict keye.g., x Image 2 Right

            

        # for index, row in file1_df.iloc[1:].iterrows(): #skip the first row because it is the header
        #     if row.notna().any(): # check if at least one cell is filled
        #         for header, value in zip(file1_df.columns, row):
        #             if "Right" in header:
        #                 if value != None: #something is detected for the right hand
        #                     if header in all_keys: #this is something we want to transfer to the new csv
        #                         d_frame[header] = value
        #                     # print(header, value)
        #     d_frame["Index"] = index
        # print(d_frame)

#for row in rows of file1
    #if not none (not includindg the index)
        #if not none in columns with right header
            #fill the new csv with the according coordinates
            #assign the file1 model to be a 1 for the right hand
    # for master_index in index_list:
    #     d_frame = {key: None for key in all_keys} #initialize new row to all None
    #     d_frame["Index"] = master_index
    #     for index, row in file1_df.iloc[1:].iterrows(): #skip the first row because it is the header
    #         if row.notna().any(): # check if at least one cell is filled
    #             for header, value in zip(file1_df.columns, row):
    #                 if "Right" in header:
    #                     if value != None: #something is detected for the right hand
    #                         if header in all_keys: #this is something we want to transfer to the new csv
    #                             d_frame[header] = value
    #                         # print(header, value)
    #         d_frame["Index"] = index
    #     print(d_frame)

#STEP 2.b: Check if right hand is full in file2. If so and 1.a false, fill with file2 and label model colummn. If 1.a true, just update model column
#STEP 2.c: (if 2.b false) Leave as empty (not detected for either) and just fill with index
#STEP 3: Repeat Step 2 with left hand

def matching_key(header, all_keys): #return the key in the dictionary that cooresponds to the header given. Necessary because of changed naming convention
    header_cor, header_num = get_pattern(header)
    for key in all_keys:
        key_cor, key_num = get_pattern(key)
        if key_cor == header_cor and key_num == header_num:
            return key

def get_pattern(string):
    cor_pattern = r'[xyz]' #x,y, or z
    num_pattern = r'(\d{1,2})' #2 digit number
    match_cor = re.findall(cor_pattern, string, re.IGNORECASE)
    match_num = re.findall(num_pattern, string, re.IGNORECASE)
    # print(match_cor, match_num)
    return (match_cor, match_num)


# matching_key("x 21 hi", ["24 x", "21 kjh x"])
file2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Holistic\0.6 Detect 0.9 Track\P01\B&B_54%_P01_B_pre_preprocessed_0.6d_0.9t_0.34%.csv"
file1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Hand\0.1d 0.5p 1.0t\P01\B&B_20_%P01_B_pre_preprocessed_0.1d_0.5p_1t_19.0%.csv"
combine_outputs(file1, file2)

        # if row.isna().all() == True: #check all cells are empty
