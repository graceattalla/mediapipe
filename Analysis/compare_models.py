#Bar graph with % frames with landmarks detected for all videos in each model.

import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style

#Plot percent frames w landmarks for all videos (single subplot can be saved)
def graph_folder_allvideos_subplot(folder):
    parent_folder = os.listdir(folder)
    fig, axs = plt.subplots(4, 1, figsize=(8, 4*4), tight_layout=True)
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

#Bar charts of number of participants with percentage of frames with landmarks above different thresholds
def threshold_subplot(folder, thres): #threshold is a list
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
        axs[i].set_title(f" Above {thres}% {file_name}")
        axs[i].set_ylabel(f"Num Above {thres}%")
        i += 1

    fig.suptitle('Num Videos Above Thresholds by Model', fontsize=16)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.title('Num Frames with Landmark Percent Above Thresholds')
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

#Loop through row of csv and determine if there is a pair.
#Determine if both of the pairs are above the threshold.
#If so, add to a counter.

def pair_threshold(file_path): #file to analyze
    data=pd.read_excel(file_path, index_col=0)
    for 


folder = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames"
# threshol_subplot(folder, [30, 40, 50, 60])
# graph_folder_allvideos_subplot(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames")
# graph_folder_avgparticipant_subplot(folder)
allvideos_model_diff(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames\Model B&B %\Hand Models")
