# note: right now only checks for left right shaking and does not account for lost nodes in old_vec to new_vec
from dotenv import load_dotenv
import numpy as np
import cv2 as cv
from testrtsp import *
import os

from functions import *

load_dotenv()

# camera = RTSPCameraStream(os.getenv("rtsp_url"))
# camera.start()

# try:
# while True:
#     ret, frame = camera.read()
#     if ret:
#         cv2.imshow("RTSP Camera Stream", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print("Failed to get frame")

# define stuff like where input is from and parameters for lk and st
cap = cv.VideoCapture(0)


st_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 10,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))


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
        if((new_vec[i][0]<0 and old_vec[i][0]>0 and old_vec[i][0]-new_vec[i][0]>1) or (new_vec[i][0]>0 and old_vec[i][0]<0 and old_vec[i][0]-new_vec[i][0]<-1)):
            counth+=1
        if((new_vec[i][1]<0 and old_vec[i][1]>0 and old_vec[i][1]-new_vec[i][1]>1) or (new_vec[i][1]>0 and old_vec[i][1]<0 and old_vec[i][1]-new_vec[i][1]<-1)):
            countv+=1
        new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
    if counth > 7*len(p1)/8 or countv > 7*len(p1)/8:
        print('shake?')
        
    cv.imshow('frame', new_frame)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('q'):
        break

    old_gray, p0= shift_data(new_gray, good_new)
    old_vec = new_vec
    
    if k == ord('r') or len(p0) < 12:
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