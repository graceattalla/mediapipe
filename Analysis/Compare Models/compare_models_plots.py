#Bar graph with % frames with landmarks detected for all videos in each model.

import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style
import plotly.express as px

#Plot percent frames w landmarks for all videos (single subplot can be saved)
def allvideos_subplot(folder):
    parent_folder = os.listdir(folder)
    num_models = len(parent_folder)
    fig, axs = plt.subplots(num_models, 1, figsize=(8, num_models*4), tight_layout=True)
    i = 0
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            file_name = os.path.splitext(file)[0]
            data=pd.read_excel(file_path, index_col=0)
            x=data.index
            y=data["Percent Frames with Landmark"]
            axs[i].bar(x,y)
            axs[i].set_title(file_name)
            i+=1

    fig.suptitle('Percent Frames w Landmarks - All Videos', fontsize=16)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title('Percent Frames w Landmarks - All Videos')
    plt.show()

#Plot percent frames w landmarks DIFFERENCE for all videos (single subplot can be saved)
def allvideos_difference_subplot(file1, file2):

    y_diff = []
    data1 = pd.read_excel(file1, index_col=0)
    data2 = pd.read_excel(file2, index_col=0)
    y_diff = data1["Percent Frames with Landmark"] - data2["Percent Frames with Landmark"]
    x_data = data1.index

    #arrange max to min
    # y_diff_sorted = y_diff.sort_values(ascending=False)
    # x_data_newindex = pd.Series(range(1, len(y_diff_sorted) + 1))
    # print(y_diff_sorted)
    diff_df = pd.DataFrame({"Index": x_data,
                   "Percent Frames with Landmark Difference": y_diff})

    sns.barplot(x='Index', y='Percent Frames with Landmark Difference', data=diff_df, color = 'dodgerblue', 
                order=diff_df.sort_values('Percent Frames with Landmark Difference', ascending=False).Index)
    plt.title("Percent Difference (Hand Landmark - Holistic) of Frames with Detected Landmark per Video")
    plt.xlabel("Participant Videos (n=200)")
    plt.ylabel("Percent Difference (Hand Landmark - Holistic) of Frames with Detected Landmark")
    plt.gca().set_xticklabels([])
    plt.show()

#Plot the percentage frames w landmark DIFFERENCE for average of each participant
def avg_participant_difference(file):
    df = pd.read_excel(file, "Hand & Holistic Wide Format")
    sns.barplot(x='Participant', y='Difference', data=df, color = 'dodgerblue', 
                order=df.sort_values('Difference', ascending=False).Participant)
    plt.title("Percent Difference (Hand Landmark Model - Holistic Model) of Avg. Frames with Detected Landmark per Participant")
    plt.xlabel("Participants (n=35)")
    plt.ylabel("Percent Difference (Hand Landmark - Holistic) of Avg. Frames with Detected Landmark")
    plt.gca().set_xticklabels([])
    plt.show()


#Plot percent frames w landmarks for all videos (need to save each graph seperately)
def graph_folder_allvideos(folder):

    parent_folder = os.listdir(folder)
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            data=pd.read_excel(file_path, index_col=0)
            fig = plt.figure(num=f'Percent Frames w Landmarks - All Videos - {file}')
            sns.barplot(x=data.index, y=data["Percent Frames with Landmark"])
            plt.title(f'Percent Frames w Landmarks - All Videos - {file}')
            plt.show()

#Plot average percent frames for each participant
def graph_folder_avgparticipant_subplot(folder):
    parent_folder = os.listdir(folder)
    fig, axs = plt.subplots(4, 1, figsize=(8, 4*4), tight_layout=True)
    i = 0
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            file_name = os.path.splitext(file)[0]
            data=pd.read_excel(file_path, index_col=0)
            x_orig=data.index
            mask = data["Average of Each Participant"].notnull()
            x = x_orig[mask]
            print(x)
            y=data.loc[mask, "Average of Each Participant"]
            axs[i].bar(x,y)
            axs[i].set_title(file_name)
            i+=1
    fig.suptitle('Percent Frames w Landmarks - Per Participant', fontsize=16)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title('Percent Frames w Landmarks - Per Participant')
    plt.show()

#Ternary plot for average percentage of participants for the 3 
def ternary(file):
    df = pd.read_excel(file, "Ternary Plot All Data")
    fig = px.scatter_ternary(df, a="Holistic", b="Hand", c="Hand Legacy")
    fig.update_layout(
        # title="Comparison of Landmark Detection Presence: Hand vs Hand Legacy vs Holistic Models",
        # title_font=dict(size=24),
        ternary_aaxis_title_font=dict(family="Calibri", size=20, color="Black"),  # Font size for axis A title
        ternary_baxis_title_font=dict(family="Calibri", size=20, color="Black"),  # Font size for axis B title
        ternary_caxis_title_font=dict(family="Calibri", size=20, color="Black"),  # Font size for axis C title
)
    fig.update_traces(marker=dict(size=12))
    fig.show()

#Bar plot of percent increase from combining models
def percent_increase(file):
    df = pd.read_excel(file)
    sns.barplot(x='File Name', y='Hand - Holistic', data=df, color = 'deepskyblue', 
                order=df.sort_values('Hand - Holistic', ascending=False)["File Name"])
    sns.barplot(x='File Name', y='Gained %', data=df, color = 'navy', 
                order=df.sort_values('Hand - Holistic', ascending=False)["File Name"])
    plt.title("Percent Difference Hand minus Holistic (Dark) & Percent Gained Per Video (dark)")
    plt.xlabel("Videos (n=200)")
    plt.ylabel("Percent")
    plt.gca().set_xticklabels([])


    plt.show()


#Bar chart of number of participants with percentage of frames with landmarks above threshold of 30%
def threshold(folder, threshold):
    parent_folder = os.listdir(folder)
    # fig, axs = plt.subplots(4, 1, figsize=(16, 3*4), tight_layout=True)
    x = []
    y = []
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            file_name = os.path.splitext(file)[0]
            data=pd.read_excel(file_path, index_col=0)
            greater = (data["Percent Frames with Landmark"] > threshold).sum()
            x.append(file_name)
            y.append(greater)
    sns.barplot(x=x, y=y)
    plt.xlabel("Model")
    plt.ylabel(f"Num Videos Greater Than {threshold}%")
    plt.title(f"Num Videos Above {threshold}% by Model")
    plt.show()

#Bar charts of number of videos with percentage of frames with landmarks above different thresholds
def threshold_multithres(folder, thres): #threshold is a list
    parent_folder = os.listdir(folder)
    fig, axs = plt.subplots(len(thres), 1, figsize=(14, len(thres)*3), tight_layout=True)
    i = 0
    for thres in thres: #iterate through each of the 4 thresholds
        x = []
        y = []
        for file in parent_folder:
            if ".xlsx" in file:
                file_path = os.path.join(folder, file)
                file_name = os.path.splitext(file)[0]
                print(file_name)
                data=pd.read_excel(file_path, index_col=0)
                greater = (data["Percent Frames with Landmark"] > thres).sum()
                x.append(file_name)
                y.append(greater)
        axs[i].bar(x,y)
        axs[i].set_title(f" Above {thres}%")
        axs[i].set_ylabel(f"Num Above {thres}%")
        i += 1

    fig.suptitle('Num Videos Above Thresholds by Model', fontsize=16)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title('Num Videos with Landmark Percent Above Thresholds')
    plt.show()

#Number of pairs of videos with pre & post both above threshold.
#To Do:
#- change to total num videos like multithres below
def pair_threshold(folder, thres):
    parent_folder = os.listdir(folder)
    x = []
    y = []
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            file_name = os.path.splitext(file)[0]
            data=pd.read_excel(file_path, index_col=0)
            prev_name = ''
            prev_name_split = [None, None, None] #initializing list of 3 for first loop
            pairs_greater = 0 #the number of pairs that have a percent above the threshold provided
            for index, row in data.iterrows(): #row is the current row
                cur_percent = row["Percent Frames with Landmark"]
                cur_name = row['Video Name'] #e.g. P01_A_post_preprocessed_0.2d_0.7t_9.0%.csv
                cur_name_split = cur_name.split('_') #e.g., ['P01', 'A', 'post', 'preprocessed', '0.2d', '0.7t', '9.0%.csv']
                #Need to deal with first iteration
                if (cur_name_split[0] == prev_name_split[0]) and (cur_name_split[1] == prev_name_split[1]): #checks if same participant and trial- if both true then must be pre and post
                    #if one is pre and one is post
                    if ((cur_percent > thres) and (prev_percent > thres)):
                        pairs_greater += 1
                prev_percent = row["Percent Frames with Landmark"]
                prev_name = cur_name
                prev_name_split = cur_name_split
            x.append(file_name)
            y.append(pairs_greater)
            print(pairs_greater)
    sns.barplot(x=x, y=y)
    plt.xlabel("Model")
    plt.ylabel(f"Num Pair Videos (Pre & Post) Greater Than {thres}%")
    plt.title(f"Num Pair Videos (Pre & Post) Above {thres}% by Model")
    plt.show()
            #need to get participant, A/B/C, pre/post. Save this in a list
            # print(row['Video Name'])

#Bar charts of number of videos (where pre/post pair is also above threshold) with percentage of frames with landmarks above different thresholds
#All files should be in the same folder
def pair_threshold_multithres(folder, thres): #threshold is a list
    parent_folder = os.listdir(folder)
    fig, axs = plt.subplots(len(thres), 1, figsize=(14, len(thres)*3), tight_layout=True)
    i = 0
    for thres in thres:
        x = []
        y = []
        for file in parent_folder:
            if ".xlsx" in file:
                file_path = os.path.join(folder, file)
                file_name = os.path.splitext(file)[0]
                data=pd.read_excel(file_path, index_col=0)
                prev_name = ''
                prev_name_split = [None, None, None] #initializing list of 3 for first loop
                vids_greater = 0 #the number of videos that have a percent above the threshold provided given both pairs above threshold
                for index, row in data.iterrows(): #row is the current row
                    cur_percent = row["Percent Frames with Landmark"]
                    cur_name = row['Video Name'] #e.g. P01_A_post_preprocessed_0.2d_0.7t_9.0%.csv
                    cur_name_split = cur_name.split('_') #e.g., ['P01', 'A', 'post', 'preprocessed', '0.2d', '0.7t', '9.0%.csv']
                    #Need to deal with first iteration
                    if (cur_name_split[0] == prev_name_split[0]) and (cur_name_split[1] == prev_name_split[1]): #checks if same participant and trial- if both true then must be pre and post
                        #if one is pre and one is post
                        if ((cur_percent > thres) and (prev_percent > thres)):
                            vids_greater += 2
                    prev_percent = row["Percent Frames with Landmark"]
                    prev_name = cur_name
                    prev_name_split = cur_name_split
                x.append(file_name)
                y.append(vids_greater)
        axs[i].bar(x,y)
        axs[i].set_title(f"Above {thres}%")
        axs[i].set_ylabel(f"Num Vids Above {thres}%")
        i += 1

    fig.suptitle('Num Videos Pre & Post Both Above Thresholds by Model', fontsize=16)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title('Num Videos Pre & Post Both with Landmark Percent Above Thresholds')
    plt.show()

#Plot difference between hand model percentage for every video.
#To Do:
# y label is too long- just take model part of file name
# x label is too squished with indicies
def allvideos_model_diff(folder): #Folder should contain 2 models to compare. Folder name should describe the models (e.g., "Hand Models")
    parent_folder = os.listdir(folder) #list of what is in folder (not full path of files in folder)
    print(parent_folder)
    folder_name = os.path.basename(os.path.normpath(folder))
    # fig, axs = plt.subplots(4, 1, figsize=(8, 4*4), tight_layout=True)
    i = 0
    y = []
    x = []
    file_names = [] #keep track of the order of this list
    for file in parent_folder:
        if ".xlsx" in file:
            file_path = os.path.join(folder, file)
            file_name = os.path.splitext(file)[0]
            data=pd.read_excel(file_path, index_col=0)
            x=data.index
            file_names.append(file_name)
            y.append(data["Percent Frames with Landmark"]) #will subtract these later

    y_diff = y[0] - y[1]
    sns.barplot(x=x, y=y_diff)
    plt.xlabel("Videos")
    plt.ylabel(f"Percent Frames with Landmarks Difference ({file_names[0]} - {file_names[1]})") #this is too long of a name right now (need to just get model name from the csv title)
    plt.title(f"Percent Frames with Landmarks Difference- {folder_name}")
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title(f"Percent Frames with Landmarks Difference- {folder_name}")
    plt.show()

#Bar chart of number of participant pairs (pre and post) both with percentage of frames with landmarks above threshold of 30%
#Assumes that the spreadsheet is in order of participant trials. E.g., 01 first with all pre and post back to back.

#To Do:
# Finish checking if pre and post are in prev and cur. Count if so.


#Loop through row of csv and determine if there is a pair.
#Determine if both of the pairs are above the threshold.
#If so, add to a counter.


# folder = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\B&B % Frames\Model B&B %\Hand & Holistic Models (copies)"
# file1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\B&B % Frames\Model B&B %\Hand Models\Hand Percentage Filled 0.1d 0.5p 1.0t Part. Avgs.xlsx"
# file2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\B&B % Frames\Model B&B %\Holistic Models\Holistic Percentage Filled 0.6d 0.9t Part. Avg.xlsx"
# allvideos_difference_subplot(file1, file2)
# threshol_subplot(folder, [30, 40, 50, 60])
# graph_folder_allvideos_subplot(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames")
# graph_folder_avgparticipant_subplot(folder)
# allvideos_model_diff(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames\Model B&B %\Hand Models")

# file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\B&B % Frames\Model B&B %\Average % Frames Per Participant.xlsx"
# ternary(file)

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Combined Percent Comparison.xlsx"
percent_increase(file)