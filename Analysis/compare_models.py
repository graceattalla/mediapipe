#Bar graph with % frames with landmarks detected for all videos in each model.

import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style

path = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames\Hand Legacy Percentage Filled 0.2d 0.7t Part. Avg.xlsx"

data=pd.read_excel(path, index_col=0)

print(data.head())

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

def threshold_subplot(folder, thres):
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

folder = r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames"
threshold_subplot(folder, [30, 40, 50, 60])
# graph_folder_allvideos_subplot(r"C:\Users\grace\Documents\Fatigue Study\Fatigue Videos\Rotated Videos\Rotated (Mediapipe)\MediaPipe Done\B&B % Frames")
# graph_folder_avgparticipant_subplot(folder)