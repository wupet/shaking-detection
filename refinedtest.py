# note: right now only checks for left right shaking and does not account for lost nodes in old_vec to new_vec
import numpy as np
import cv2 as cv

# def get_proc_next_frame()
#     #get next frame
#     ret, new_frame = cap.read()
#     if not ret:
#         print("Cannot read video file")
#         exit()
#     # process next frame
#     new_frame = cv.flip(new_frame,1)
#     return new_frame,cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

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

# get first frame
ret, old_frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()
# process first frame
old_frame = cv.flip(old_frame,1)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **st_params)

# get second frame
ret, new_frame = cap.read()
if not ret:
    print("Cannot read video file")
    exit()
# process second frame
new_frame = cv.flip(new_frame,1)
new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

# calculate new points
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

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
print(old_vec)
# show the new frame
cv.imshow('frame', new_frame)

# shift new data to old 
old_gray = new_gray.copy()
p0 = good_new.reshape(-1, 1, 2)

while(True):
    ret, new_frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    new_frame = cv.flip(new_frame,1)
    new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    new_vec=[]
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    count = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        new_vec.append([a-c, b-d])
        if((new_vec[i][0]<0 and old_vec[i][0]>0 and old_vec[i][0]-new_vec[i][0]>1) or (new_vec[i][0]>0 and old_vec[i][0]<0 and old_vec[i][0]-new_vec[i][0]<-1)):
            count+=1
        new_frame = cv.circle(new_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    if count > len(p1)/2:
        print('shake?')
    cv.imshow('frame', new_frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('q'):
        break

    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    old_vec = new_vec
    
    if k == ord('r'):
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
        print('reset')
        # get second frame
        ret, new_frame = cap.read()
        if not ret:
            print("Cannot read video file")
            exit()
        # process second frame
        new_frame = cv.flip(new_frame,1)
        new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # calculate new points
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

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
        print(old_vec)
        # show the new frame
        cv.imshow('frame', new_frame)

        # shift new data to old 
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    if len(p0) < 12:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
        print('reset')
        # get second frame
        ret, new_frame = cap.read()
        if not ret:
            print("Cannot read video file")
            exit()
        # process second frame
        new_frame = cv.flip(new_frame,1)
        new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # calculate new points
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

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
        print(old_vec)
        # show the new frame
        cv.imshow('frame', new_frame)

        # shift new data to old 
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()