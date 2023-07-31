import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style
import pingouin as pg


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

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\B&B % Frames\Model B&B %\Average % Participants.xlsx"

df = pd.read_excel(file, "Sheet2")

aov = pg.anova(dv='Percentage', between='Model', data=df, detailed=True)

pg.print_table(aov)

#run ANOVA on average percentage for each participant across models

#run pairwise on average percentage for each participant acros models