# note: right now does not account for lost nodes in old_vec to new_vec
# work on installing ffmpeg and using it to stream the rtsp since opencv is too slow.
# consider checking all points for similar vector movement insteead of vector movement in general
# we can change various tuning by creating 2 sets of data and shifting until we get there. something something ai training
from dotenv import load_dotenv
import numpy as np
import cv2 as cv
from irrelevant_tests.testrtsp import *
import os

from functions import *

load_dotenv()

# camera = RTSPCameraStream(os.getenv("rtsp_url"))
# camera.start()

# # try:
# while True:
#     ret, frame = camera.read()
#     if ret:
#         cv2.imshow("RTSP Camera Stream", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print("Failed to get frame")

# define stuff like where input is from and parameters for lk and st
cap = cv.VideoCapture("shaking\\3967271-uhd_4096_2160_24fps.mp4")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(width)
print(height)
    
# magnitude of shaking needed to set off the alarm
sensitivity = min(width, height)/720
# ratio of points shaking to set off alarm
sensitivity2 = 7/8
# how spread apart the points should be i.e. percentage 
spread = min(width, height)/50

st_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = spread,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



color = np.random.randint(0, 255, (100, 3))

# --- DYNAMIC ADAPTIVE WINDOW SETUP ---
# 1. Get original video dimensions
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print(width)
# print(height)
# # 2. Define a maximum bounding box for the window (e.g., fits comfortably on a 1080p screen)
MAX_WIDTH = 960
MAX_HEIGHT = 540

# # 3. Calculate scaling factor to preserve aspect ratio
scale = min(MAX_WIDTH / width, MAX_HEIGHT / height)
window_w = int(width * scale)
window_h = int(height * scale)

# # 4. Initialize the window with the dynamic dimensions
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', window_w, window_h)
# # -------------------------------------

old_frame, old_gray = get_proc_next_frame(cap)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **st_params)

new_frame, new_gray = get_proc_next_frame(cap)
# calculate new points
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

good_new, old_vec = processpts(p0, p1, new_frame, color, st)

# show the new frame
cv.imshow('frame', new_frame)

# shift new data to old 
old_gray, p0 = shift_data(new_gray, good_new)

while(True):
    new_frame, new_gray = get_proc_next_frame(cap)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    new_vec=[]
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    counth = 0
    countv = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        new_vec.append([a-c, b-d])
        if((new_vec[i][0]<0 and old_vec[i][0]>0 and old_vec[i][0]-new_vec[i][0]>sensitivity) or (new_vec[i][0]>0 and old_vec[i][0]<0 and old_vec[i][0]-new_vec[i][0]<sensitivity)):
            counth+=1
        if((new_vec[i][1]<0 and old_vec[i][1]>0 and old_vec[i][1]-new_vec[i][1]>sensitivity) or (new_vec[i][1]>0 and old_vec[i][1]<0 and old_vec[i][1]-new_vec[i][1]<sensitivity)):
            countv+=1
        new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
    if counth > sensitivity2*len(p1) or countv > sensitivity2*len(p1):
        print('shake?\n')
        
    cv.imshow('frame', new_frame)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('q'):
        break

    old_gray, p0= shift_data(new_gray, good_new)
    old_vec = new_vec
    
    if k == ord('r') or len(p0) < 10:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
        print('reset')
        new_frame, new_gray = get_proc_next_frame(cap)

        # calculate new points
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        good_new, old_vec = processpts(p0, p1, new_frame, color, st)
        
        # show the new frame
        cv.imshow('frame', new_frame)

        old_gray, p0 = shift_data(new_gray, good_new)
    
cv.destroyAllWindows()