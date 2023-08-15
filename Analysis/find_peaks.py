'''
Find peaks from y coordinates.
'''

from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file = r"C:\Users\grace\OneDrive\BCI4Kids (One Drive)\MediaPipe Done\Combine Models\Pre&Post Greater 60%\B\P29\barrier_combined_B&B_67%_P29_B_pre_preprocessed.csv"

file_name = os.path.basename(file)
file_name = os.path.splitext(file_name)[0]

folder_path = os.path.dirname(file)

df = pd.read_csv(file)
x = df["y Image 5 Right barrier total norm"]
peaks, peak_info = find_peaks(x, height=2, distance=40)
mean_peak_height =x[peaks].mean()
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()
peaks_df = pd.DataFrame({"Index": peaks,"Peaks": x[peaks], "Mean": mean_peak_height})

print(peaks_df)

save_path = folder_path + "\\" + "peaks_" + file_name + ".csv"
peaks_df.to_csv(save_path)
