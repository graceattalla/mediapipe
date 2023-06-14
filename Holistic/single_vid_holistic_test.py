import mediapipe_process_holistic as p
import numpy as np
import time

start_time = time.time()


VIDEO = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Holistic\Fatigue Cropped\P05\Full BB Video\P05_B_post_cropped_full_1080p.mp4"

# for value in np.arange(0.3, 1.0, 0.1):
p.process_video(VIDEO)

end_time = time.time()
execution_time = end_time - start_time

print("Execution Time:", execution_time, "seconds")

#11:02- approx 11:14
#11:35
