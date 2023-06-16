import cv2
import numpy as np
from matplotlib import pyplot as plt

video_to_process = r"C:\Users\grace\OneDrive\Surface Laptop Desktop\BCI4Kids\Mediapipe\Videos\Fatigue_Vids_Original\P19\P19_B_post.mp4"

cap = cv2.VideoCapture(video_to_process)
    
#Deal with writing a video out from CV2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# while(cap.isOpened()):
#Capture each frame-by-frame
frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) #frame number must be an int I think

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

ret, frame = cap.read()

# reading image
shape_img = np.copy(frame)

# converting image into grayscale image
gray = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

#define desired shape
desired_width_height_ratio = 10/3

# list for storing names of shapes
for contour in contours:

	# here we are ignoring first counter because
	# findcontour function detects whole image as shape
	if i == 0:
		i = 1
		continue

	# cv2.approxPloyDP() function to approximate the shape
	approx = cv2.approxPolyDP(
		curve = contour, epsilon = 0.01 * cv2.arcLength(contour, True), closed=True)
	
	x, y, w, h = cv2.boundingRect(approx)
	width_height_ratio = w/h
	tolerance = 3
	if abs(width_height_ratio - desired_width_height_ratio) < tolerance:
		# Shape of desired dimensions found, do something with it
		cv2.rectangle(shape_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# using drawContours() function
	# cv2.drawContours(shape_img, [contour], 0, (0, 0, 255), 5)

	# finding center point of shape
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])

	# putting shape name at center of each shape
	# if len(approx) == 3:
	# 	cv2.putText(shape_img, 'Triangle', (x, y),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	if len(approx) == 4:
		cv2.putText(shape_img, 'Quadrilateral', (x, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	# elif len(approx) == 5:
	# 	cv2.putText(shape_img, 'Pentagon', (x, y),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	# elif len(approx) == 6:
	# 	cv2.putText(shape_img, 'Hexagon', (x, y),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	# else:
	# 	cv2.putText(shape_img, 'circle', (x, y),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# displaying the image after drawing contours
cv2.imshow('shapes', shape_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
