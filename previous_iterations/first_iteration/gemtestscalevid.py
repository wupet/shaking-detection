# note: right now does not account for lost nodes in old_vec to new_vec
# work on installing ffmpeg and using it to stream the rtsp since opencv is too slow.
# consider checking all points for similar vector movement insteead of vector movement in general
# we can change various tuning by creating 2 sets of data and shifting until we get there. something something ai training
# issue with p1 being none when no points to track
from dotenv import load_dotenv
import numpy as np
import cv2 as cv
from previous_iterations.irrelevant_tests.testrtsp import *
import os

from functions import *

load_dotenv() # 111 notshaking 34 shaking

# define stuff like where input is from and parameters for lk and st
cap = cv.VideoCapture("not shaking\\notshaking1.mp4")

# --- RESOLUTION TARGETING SETUP ---
# Define a standard fixed processing width


# Read original dimensions to compute the correct target height dynamically
orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
TARGET_WIDTH = 640
if TARGET_WIDTH < orig_width:
    scale_factor = TARGET_WIDTH / orig_width
    TARGET_HEIGHT = int(orig_height * scale_factor)
else:
    TARGET_HEIGHT = orig_height
    TARGET_WIDTH = orig_width

print(f"Original Resolution: {orig_width}x{orig_height}")
print(f"Downsampled Processing Resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
# ----------------------------------

# Helper function to grab and instantly resize raw frames
def get_scaled_frame(capture_device, target_w, target_h):
    frame, gray = get_proc_next_frame(capture_device)
    scaled_frame = cv.resize(frame, (target_w, target_h))
    scaled_gray = cv.resize(gray, (target_w, target_h))
    return scaled_frame, scaled_gray

# Tuning variables are static because they are calibrated for our 960x540 canvas size
sensitivity = 1    # magnitude of shaking looked for (higher = less shaking detected)
qopts = .1         # quality of points selected (lower = more points)
sensitivity2 = .9 # ratio of points needed to shake (higher = less shaking detected)
spread = 10      # minimum distance between points when initialized (higher = more spreadout, more resets)
minpts = 10        # minimum number of points tracked (higher = more resets, potentially better data)

st_params = dict( maxCorners = 50,
                       qualityLevel = qopts,
                       minDistance = int(spread),
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

# Initialize the window matching our target processing dimensions exactly
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', TARGET_WIDTH, TARGET_HEIGHT)

# Initial frame setups using the downscaler helper
old_frame, old_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **st_params)

new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
# calculate new points
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

good_new, old_vec = processpts(p0, p1, new_frame, color, st)

# show the new frame
cv.imshow('frame', new_frame)

# shift new data to old 
old_gray, p0 = shift_data(new_gray, good_new)

while(True):
    # Downscale the frames immediately at the start of every loop iteration
    new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
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
        if((new_vec[i][0]<0 and old_vec[i][0]>0 and old_vec[i][0]-new_vec[i][0]>sensitivity) or 
           (new_vec[i][0]>0 and old_vec[i][0]<0 and new_vec[i][0]-old_vec[i][0]>sensitivity)):
            counth+=1
        if((new_vec[i][1]<0 and old_vec[i][1]>0 and old_vec[i][1]-new_vec[i][1]>sensitivity) or 
           (new_vec[i][1]>0 and old_vec[i][1]<0 and new_vec[i][1]-old_vec[i][1]>sensitivity)):
            countv+=1
        new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
    if counth > sensitivity2*len(p1) or countv > sensitivity2*len(p1):
        print('shake?')
        
    cv.imshow('frame', new_frame)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('q'):
        break

    old_gray, p0= shift_data(new_gray, good_new)
    old_vec = new_vec
    
    if k == ord('r') or len(p0) < minpts:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
        print('reset')
        new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)

        # calculate new points
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        good_new, old_vec = processpts(p0, p1, new_frame, color, st)
        
        # show the new frame
        cv.imshow('frame', new_frame)

        old_gray, p0 = shift_data(new_gray, good_new)
    
cv.destroyAllWindows()