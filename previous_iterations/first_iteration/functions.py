import numpy as np
import cv2 as cv

def shift_data(new_gray, good_new, new_vec):
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return old_gray, p0, new_vec

def shift_data(new_gray, good_new):
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return old_gray, p0

def get_proc_next_frame(cap):
    #get next frame
    ret, new_frame = cap.read()
    if not ret:
        print("Cannot read video file")
        exit()
    # process next frame
    # new_frame = cv.flip(new_frame,1)
    return new_frame,cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

def processpts(p0,p1,new_frame,color,st):
    # if p1 is valid, we put all points from the first two frames  
    # that are still found in second frame from the first frame into
    # good_new and good_old 
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # harvest the two sets of points into x and y values and draws
    # a point at the new coordinates
    # a = newx, b = newy, c = oldx, d = oldy
    # also, we create a list of vectors that describe the change from new to old
    old_vec = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        old_vec.append([a-c, b-d])
        new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    # print(old_vec)
    return good_new, old_vec

# def reset():
#     p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
#     print('reset')
#     # get second frame
#     ret, new_frame = cap.read()
#     if not ret:
#         print("Cannot read video file")
#         exit()
#     # process second frame
#     new_frame = cv.flip(new_frame,1)
#     new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

#     # calculate new points
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

#     # if p1 is valid, we put all points from the first two frames  
#     # that are still found in second frame from the first frame into
#     # good_new and good_old 
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]

#     # harvest the two sets of points into x and y values and draws
#     # a point at the new coordinates
#     # a = newx, b = newy, c = oldx, d = oldy
#     # also, we create a list of vectors that describe the change from new to old
#     old_vec = []
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         old_vec.append([a-c, b-d])
#         new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
#     print(old_vec)
#     # show the new frame
#     cv.imshow('frame', new_frame)

#     # shift new data to old 
#     old_gray = new_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)
#     return p0, 