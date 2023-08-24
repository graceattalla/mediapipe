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

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Combined Percent Comparison.xlsx"

df = pd.read_excel(file)
print(df.head())

#Check for Normality (use wide format)
normality = pg.normality(df["Gained %"], method='normaltest')
pg.print_table(normality)

# #Quantile-quantile plot
# data = df["Hand Legacy Avg Percent Frames with Detected Landmark"]
# qq = pg.qqplot(data, dist = 'norm')
# plt.show()

# # Find the outlier points
# outliers = data[qq["outliers"]]
# print(outliers)

#One way ANOVA (use long format)
# aov = pg.anova(dv='Percentage', within='Model', data=df, detailed=True)

# One way repeated ANOVA (use long format)
# aov = pg.rm_anova(dv='Percentage', within='Model', subject = "Participant", data=df, detailed=True)
# pg.print_table(aov)

# One-sample T-test
ttest = pg.ttest(df['Gained %'], 0) #compare with 0
pg.print_table(ttest)

#Pair-wise Test
# pair_ttest = pg.pairwise_tests(dv='Percentage', within='Model', subject = "Video Index", data=df)
# pg.print_table(pair_ttest)

# ax = pg.plot_paired(data=df, dv='Percentage', within='Model', subject='Video Index')
# ax.set_title("hand vs holistic")
# plt.show()

#Wilcoxan Signed-Rank
# wx = pg.wilcoxon(df["Percentage Hand"], df["Percentage Holistic"])
# pg.print_table(wx)
