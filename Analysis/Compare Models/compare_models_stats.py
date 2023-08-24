'''
Stats for MediaPipe model comparison. Trying to determine if two models perform differently for different videos (rather than overall overage being different)

Uses a file with the summary of the model output % frames with detected landmark csv ("Combined Percent Comparison.csv").

Written by Grace Attalla
'''

import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style
import pingouin as pg

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Combined Percent Comparison.xlsx"

df = pd.read_excel(file)
print(df.head())

#Check for Normality (use wide format)
normality = pg.normality(df["Gained %"], method='normaltest')
pg.print_table(normality)

# #Quantile-quantile plot
data = df["Hand Legacy Avg Percent Frames with Detected Landmark"]
qq = pg.qqplot(data, dist = 'norm')
plt.show()

# # Find the outlier points
outliers = data[qq["outliers"]]
print(outliers)

#One way ANOVA (use long format)
aov = pg.anova(dv='Percentage', within='Model', data=df, detailed=True)

# One way repeated ANOVA (use long format)
aov = pg.rm_anova(dv='Percentage', within='Model', subject = "Participant", data=df, detailed=True)
pg.print_table(aov)

# One-sample T-test
ttest = pg.ttest(df['Gained %'], 0) #compare with 0
pg.print_table(ttest)

#Pair-wise Test
pair_ttest = pg.pairwise_tests(dv='Percentage', within='Model', subject = "Video Index", data=df)
pg.print_table(pair_ttest)

ax = pg.plot_paired(data=df, dv='Percentage', within='Model', subject='Video Index')
ax.set_title("hand vs holistic")
plt.show()

#Wilcoxan Signed-Rank
wx = pg.wilcoxon(df["Percentage Hand"], df["Percentage Holistic"])
pg.print_table(wx)
