'''
Combine the outputs of two models to try to fill the gaps of missing data for each frame.
Input: CSV files of outputs for videos with 2 different models
Output: New CSV file for each video combining the model outputs. Columns will also include which model was used.

Note that hand and holistic label right/left hand oppositely and had to be adjusted accordingly.

Hand model is prioritized over holistic model when both have landmarks for a given frame.

Written by Grace Attalla
'''

import pandas as pd
import os
import re

#run for folder
def combine_outputs_folder(folder1, folder2, combined_save_path): #these should be the parent folders for two different models. Assume folder1 will take preference if data for frame in both.
    parent_folder1 = os.listdir(folder1)
    parent_folder2 = os.listdir(folder2)
    df_list = []
    for (inner1, inner2) in zip(parent_folder1, parent_folder2): #inner folders are participant folders. Assumed participant folders in same order for both folders.
        participant1 = os.path.basename(inner1)
        participant2 = os.path.basename(inner2)
        if inner1 == inner2: #extra check if using same participant folder
            combined_full_save_path = os.path.join(combined_save_path, inner1)
            os.makedirs(combined_full_save_path)
            
            inner1_full = os.path.join(folder1, inner1) #full path
            inner2_full = os.path.join(folder2, inner2)
            if os.path.isdir(inner1_full): #check if it is a folder
                inner_files1 = os.listdir(inner1_full) #list of files in inner participant folder
                inner_files1_BB = [filename for filename in inner_files1 if "B&B" in filename]#list of files for one participant with B&B in file (files to combine)
            if os.path.isdir(inner2_full):
                inner_files2 = os.listdir(inner2_full)
                inner_files2_BB = [filename for filename in inner_files2 if "B&B" in filename]#list of files for one participant

            for file1 in inner_files1_BB: #iterate through all files
                for file2 in inner_files2_BB:
                    if matching_key_file(file1, file2): #files are of same participant and trial (e.g., P01_A_pre)
                        file1_path = os.path.join(inner1_full, file1)
                        file2_path = os.path.join(inner2_full, file2)
                        # print(file1_path, file2_path)
                        output = combine_outputs(file1_path, file2_path, participant1, combined_full_save_path)
                        base_name = output[0]
                        new_percent = output[1] #percentage of combined rows filled
                        percent1 = output[2] #original percentage rows filled of first file
                        percent2 = output[3]

                        d_file = {}
                        d_file["File Name"] = base_name
                        d_file["Combined Model %"] = new_percent
                        d_file["Model 1 %"] = percent1
                        d_file["Model 2 %"] = percent2

                        gained_percent = int(float(new_percent) - max(float(percent1), float(percent2)))
                        d_file["Gained %"] = gained_percent

                        df_list.append(d_file)
    output_df = pd.DataFrame(df_list)
    output_df.to_csv(combined_save_path + "\Combined Percent Comparison.csv")

                        #need to save the filename, percent of new, percent of 2 olds. Need to return percents from the function then 


def combine_outputs(file1, file2, participant, combined_full_save_path): #two files for same video but from different models. Assumes 1 is hand and 2 is holistic

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
    #STEP 2.b: Check if right hand is full in file2. If so and 1.a false, fill with file2 and label model colummn. If 1.a true, just update model column
    #STEP 2.c: (if 2.b false) Leave as empty (not detected for either) and just fill with index
    #STEP 3: Repeat Step 2 with left hand

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

                    if d_frame[match_key] == None: #has not been filled already by file 1
                        d_frame[match_key] = value
                else: #nothing detected
                    d_frame["Holistic Left"] = 0

            if "Left" in header:
                if pd.notna(value): #the second model has a value
                    match_key = matching_key_swap(header, all_keys) #not exact matches between each model so need to check
                    d_frame["Holistic Right"] = 1

                    if d_frame[match_key] == None: #has not been filled already by file 1
                        d_frame[match_key] = value

                else: #nothing detected
                    d_frame["Holistic Right"] = 0
        df_list.append(d_frame)
        # print(master_index)
    output_df = pd.DataFrame(df_list)

    non_none_rows = output_df.iloc[:, 6:].notna().any(axis=1).sum()
    percent_filled = (round(non_none_rows/output_df.shape[0], 2))*100
    print(f"non-None row: {non_none_rows}")
    print(f"% data rows: {percent_filled}")

    base_name = extract_file_name(file1)

    #find the percent of rows filled by the original models.
    non_none_rows1 = file1_df.iloc[:, 2:].notna().any(axis=1).sum()
    percent1 = (round(non_none_rows1/file1_df.shape[0], 2))*100

    non_none_rows2 = file2_df.iloc[:, 2:].notna().any(axis=1).sum()
    percent2 = (round(non_none_rows2/file2_df.shape[0], 2))*100

    print(percent1, percent2)

    # percent1 = extract_percent(file1) #percentage of the rows filled in the first file
    # percent2 = extract_percent(file2)

    save_path = combined_full_save_path + f"\combined_B&B_{int(percent_filled)}%_{base_name}.csv" #save to new folder with participant ID folder
    output_df.to_csv(save_path)

    return (base_name, percent_filled, percent1, percent2)

#return the key in the dictionary that cooresponds to the header given. Necessary because of changed naming convention
def matching_key(header, all_keys):
    header_cor, header_num, header_hand = get_pattern_cor(header)
    for key in all_keys:
        key_cor, key_num, key_hand = get_pattern_cor(key)
        if key_cor == header_cor and key_num == header_num and header_hand == key_hand:
            return key

#return the key in the dictionary that cooresponds to the header given. If the header is left then use the right dictionary label (swapped labelling convention of holistic model)
def matching_key_swap(header, all_keys): 
    header_cor, header_num, header_hand = get_pattern_cor(header)
    for key in all_keys:
        key_cor, key_num, key_hand = get_pattern_cor(key)
        if key_cor == header_cor and key_num == header_num and header_hand != key_hand: #swap left/right
            return key

def get_pattern_cor(string):
    cor_pattern = r'[xyz]' #x,y, or z
    num_pattern = r'(\d{1,2})' #2 digit number
    hand_pattern = r'(Left|Right)'  # Left or Right
    match_cor = re.findall(cor_pattern, string, re.IGNORECASE)
    match_num = re.findall(num_pattern, string, re.IGNORECASE)
    match_hand = re.findall(hand_pattern, string, re.IGNORECASE)
    # print(match_cor, match_num)
    return (match_cor, match_num, match_hand)

#Get the file name without percentages and model specific values
def extract_file_name(filepath): #e.g., P01_B_pre_preprocessed
    file_name = os.path.basename(filepath) #filename without path
    pattern1 = r'(P\d+_[A-Za-z]+_(pre|post)+_preprocessed)' #REMOVE "preprocessed". Some have another word before that.
    match = re.search(pattern1, file_name)
    if match:
        print(f"match: {match.group(1)}")
        return match.group(1)

#return true if file1 and file2 are of same participant and trial
def matching_key_file(file1, file2): 
    print(file1, file2)
    file1_id = get_pattern_file(file1)
    file2_id = get_pattern_file(file2)
    if file1_id == file2_id:
        return True

#returns the identifying information of the file (e.g., P01_C_pre)
def get_pattern_file(file_name): 
    pattern = r"(P\d+_[A-Za-z]+_(?:post|pre))"
    match = re.search(pattern, file_name)
    if match:
        print(match.group(1))
        return match.group(1)

#extract the percentage of filled rows from the file name (two different naming conventions)
#this function would be better replaced by just searching for the percent in the file.
def extract_percent(file_name): 

    
    pattern2 = r"(_)(\d+)(_)"
    match2 = re.search(pattern2, file_name)
    if match2:
        print(f"match2: {match2.group(2)}")
        return match2.group(2)

    pattern4 = r"(_)(\d+)(%_)" #e.g., B&B_0%_P04_A_post_preprocessed_0.6d_0.9t_0.0%.csv
    match4 = re.search(pattern4, file_name)
    if match4:
        print(f"match4: {match4.group(2)}")
        return match4.group(2)

save_path = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models"
folder1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Hand\0.1d 0.5p 1.0t"
folder2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Holistic\0.6 Detect 0.9 Track"
model1_name = "Hand 0.1d 0.5p 1.0t"
model2_name = "Holistic 0.6d 0.9t"

combine_outputs_folder(folder1, folder2, save_path)
