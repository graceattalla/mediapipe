'''
Run stats on the frames per oscillation (defined by peaks) as a way to initially test for velicyt differences.

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

file1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\B\P03\peaks_barrier_combined_B&B_63%_P03_B_pre_preprocessed.csv"
file2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\B\P03\peaks_barrier_combined_B&B_80%_P03_B_post_preprocessed.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Remove rows with None values from df1
df1 = df1.dropna()

# Remove rows with None values from df2
df2 = df2.dropna()

#Check for Normality (use wide format)
normality = pg.normality(df2["Frames per half oscillation"], method='normaltest')
pg.print_table(normality)

#Un-paired two sample ttest
ttest = pg.ttest(df1["Frames per half oscillation"], df2["Frames per half oscillation"]) #compare with 0
pg.print_table(ttest)

#Mann Whitney U test
mw = pg.mwu(df1["Frames per half oscillation"], df2["Frames per half oscillation"] )
pg.print_table(mw)