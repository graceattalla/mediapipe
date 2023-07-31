#Bar graph with % frames with landmarks detected for all videos in each model.

import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style

#Bar charts of number of videos (where pre/post pair is also above threshold) with percentage of frames with landmarks above different thresholds
#All files should be in the same folder
def pair_threshold_multithres(file, thres): #threshold is a list
    fig, axs = plt.subplots(len(thres), 1, figsize=(14, len(thres)*3), tight_layout=True)
    i = 0
    for thres in thres:
        x = []
        y = []
       
        file_name = os.path.splitext(file)[0]
        data=pd.read_csv(file, index_col=0)
        vids_greater = 0 #the number of videos that have a percent above the threshold provided given both pairs above threshold
        for index1, row1 in data.iterrows(): #This is an inefficient way of doing things... It is organized by participant
            for index2, row2 in data.iterrows():
                cur_percent1 = row1["Combined Model %"]
                cur_name1 = row1['File Name'] #e.g. P01_A_post_preprocessed_0.2d_0.7t_9.0%.csv
                cur_name_split1 = cur_name1.split('_') #e.g., ['P01', 'A', 'post', 'preprocessed', '0.2d', '0.7t', '9.0%.csv']
                
                cur_percent2 = row2["Combined Model %"]
                cur_name2 = row2['File Name'] #e.g. P02_A_post_preprocessed_0.2d_0.7t_9.0%.csv
                cur_name_split2 = cur_name2.split('_') #e.g., ['P01', 'A', 'post', 'preprocessed', '0.2d', '0.7t', '9.0%.csv']
                #Need to deal with first iteration
                if (cur_name_split1[0] == cur_name_split2[0]) and (cur_name_split1[1] == cur_name_split2[1]) and cur_name_split1[2] != cur_name_split2[2]: #checks if same participant and trial- if both true then must be pre and post
                    #if one is pre and one is post
                    if ((cur_percent1 > thres) and (cur_percent2 > thres)):
                        vids_greater += 1 #will find match twice.
                        print(cur_name_split1[0], cur_name_split2[0])
                        print(cur_name_split1[1], cur_name_split2[1])
                        print(thres)
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

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Combined Percent Comparison.csv"
pair_threshold_multithres(file, [60, 70])

#Plot difference between hand model percentage for every video.
#To Do:
# y label is too long- just take model part of file name
# x label is too squished with indicies
# def allvideos_model_diff(folder): #Folder should contain 2 models to compare. Folder name should describe the models (e.g., "Hand Models")
#     parent_folder = os.listdir(folder) #list of what is in folder (not full path of files in folder)
#     print(parent_folder)
#     folder_name = os.path.basename(os.path.normpath(folder))
#     # fig, axs = plt.subplots(4, 1, figsize=(8, 4*4), tight_layout=True)
#     i = 0
#     y = []
#     x = []
#     file_names = [] #keep track of the order of this list
#     for file in parent_folder:
#         if ".xlsx" in file:
#             file_path = os.path.join(folder, file)
#             file_name = os.path.splitext(file)[0]
#             data=pd.read_excel(file_path, index_col=0)
#             x=data.index
#             file_names.append(file_name)
#             y.append(data["Percent Frames with Landmark"]) #will subtract these later

#     y_diff = y[0] - y[1]
#     sns.barplot(x=x, y=y_diff)
#     plt.xlabel("Videos")
#     plt.ylabel(f"Percent Frames with Landmarks Difference ({file_names[0]} - {file_names[1]})") #this is too long of a name right now (need to just get model name from the csv title)
#     plt.title(f"Percent Frames with Landmarks Difference- {folder_name}")
#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.window.title(f"Percent Frames with Landmarks Difference- {folder_name}")
#     plt.show()