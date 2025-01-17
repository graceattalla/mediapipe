import pandas as pd
import os

'''

Select and save csv of only Box and Blocks data from mediapipe outputs.

CSV of start and stop times was collected by hand.

An additional CSV file is saved of the percentage of frames with landmark detection in each video.

Written by Grace Attalla

'''

'''
Plan:
For single csv
1. Import csv as dataframe
2. Import start and stop times of variable csv with same name as file
3. Edit dataframe to only go from start and stop times
4. Save percentage of non none rows to additional csv file
5. Save new dataframe as csv
*repeat for all csv files in folders.

'''
def process_folder(folder):
  
    #import the go and stop times csv as a data frame
    go_stop_path = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\Go and Stop Times- Original Videos.xlsx" #this should be manually entered
    times_df = pd.read_excel(go_stop_path)

    parent_folder = os.listdir(folder)
    percentage_list = []

    for inner in parent_folder:

        inner_folder = os.path.join(folder, inner) #get full path to inner folder

        if os.path.isdir(inner_folder): #check if it is a folder
            inner_files = os.listdir(inner_folder)
            total_percent_participant = 0
            num_files = 0

            for file in inner_files:
                if ".csv" in file:
                    if "B&B" not in file:
                        full_file_path = os.path.join(inner_folder, file)
                        percent_filled = BB_extraction(full_file_path, file, inner_folder, percentage_list, times_df)
                        total_percent_participant += percent_filled
                        num_files +=1
                        # full_file_paths = [os.path.join(inner_folder, file) for file in inner_files if ".csv" in file] #get a list of all the mediapipe csv files
                        # sheet_files.extend(full_file_paths)
            average_participant_percent = total_percent_participant/num_files
            #figure out a way to include average participant percent in the spreadsheet...
                

    percentage_df = pd.DataFrame(percentage_list)
    avg_percentage = percentage_df['Percent Frames with Landmark'].mean()
    percentage_df.loc[0, "Average Percent Frames with Landmark"] = avg_percentage
    save_path = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\Hand\0.1d 0.5p 0.7t\Hand Percentage Filled 0.1d 0.5p 0.7t.csv"
    #Save to csv file (can open in Excel)
    percentage_df.to_csv(save_path)

percentage_list = []


def BB_extraction(csv_to_process, file_name, inner_folder, percentage_list, times_df):

    #save the go and stop times of the video for the mediapipe csv
    #need to select from "Video Name" column of go and stop df such that it is included in the csv_to_process path
    desired_row = times_df[times_df['Video Name'].apply(lambda x: x in csv_to_process)].iloc[0]

    #save start and stop frames
    start_time = desired_row['"Go" Time (s)']
    stop_time = desired_row['"Stop" Time (s)']
    fps = 30

    start_frame = int(start_time*fps + 1) #+1 accounts for column names
    stop_frame = int(stop_time*fps + 1) 

    #import the mediapipe csv as a data frame
    mediapipe_df = pd.read_csv(csv_to_process, index_col=0)

    #modify dataframe to just include between start and stop frame
    bb_mediapipe_df = mediapipe_df.iloc[start_frame - 1:stop_frame] #-1 to be inclusive of start_frame

    #save percentage of rows with data to a csv

    tot_rows = bb_mediapipe_df.shape[0]
    non_none_rows =bb_mediapipe_df.iloc[:, 1:].notna().any(axis=1).sum() #don't include the index
    percent_filled = (round(non_none_rows/tot_rows, 4))*100
    if percent_filled == None:
        percent_filled = 0

    print(f"% filled: {percent_filled}")

    save_path = inner_folder + "\\" + f"B&B_{int(percent_filled)}%_" + file_name
    
    bb_mediapipe_df.to_csv(save_path)

    #create dictionary for given video csv
    video_dict = {'Video Name': file_name, 'Percent Frames with Landmark': percent_filled, 'Total Frames': tot_rows, 'Frames with Landmark': non_none_rows}
    #append to list of all percentages
    percentage_list.append(video_dict)
    return percent_filled #use this to calculate average for each participant

process_folder(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\Hand\0.1d 0.5p 0.7t")