'''
Run stats on peak amplitudes before and after BCI use.
Use find_peaks.py to get the peaks.

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

file1 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\B\P29\peaks_barrier_combined_B&B_67%_P29_B_pre_preprocessed.csv"
file2 = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\B\P29\peaks_barrier_combined_B&B_75%_P29_B_post_preprocessed.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

#Check for Normality (use wide format)
normality = pg.normality(df1["Peaks"], method='normaltest')
pg.print_table(normality)
# normality = pg.normality(df2["Peaks"], method='normaltest')
# pg.print_table(normality)

#Un-paired two sample ttest
ttest = pg.ttest(df1["Peaks"], df2["Peaks"]) #compare with 0
pg.print_table(ttest)