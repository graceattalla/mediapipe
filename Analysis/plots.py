import pandas as pd
pd.plotting.register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib import style

file_path = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\P20\combined_B&B_69%_P20_C_pre_preprocessed.csv"

data=pd.read_csv(file_path, index_col=0)

data_subset = data.iloc[1600:1700]

x_data_sub=data_subset["x Image 15 Right"]
y_data_sub=data_subset["y Image 15 Right"]
index_sub = data_subset["Index"]

x_data=data["x Image 15 Right"]
y_data=data["y Image 15 Right"]
y_data_inversed = 1-y_data
index = data["Index"]

# sns.scatterplot(x=x, y=y)
# hue = data["Index"]

plt.scatter(x=x_data, y=y_data_inversed, alpha = .8, c = index, cmap = 'seismic')
cbar = plt.colorbar()

plt.show()
