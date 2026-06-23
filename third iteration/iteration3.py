# note: right now does not account for lost nodes in old_vec to new_vec
# work on installing ffmpeg and using it to stream the rtsp since opencv is too slow.
# consider checking all points for similar vector movement insteead of vector movement in general
import numpy as np
import cv2 as cv
from irrelevant_tests.testrtsp import *
from functions import *

# define stuff like where input is from and parameters for lk and st
cap = cv.VideoCapture("shaking\\shaking12.mp4")

# --- RESOLUTION TARGETING SETUP ---
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
    if frame is None:
        exit()
    scaled_frame = cv.resize(frame, (target_w, target_h))
    scaled_gray = cv.resize(gray, (target_w, target_h))
    return scaled_frame, scaled_gray

# Tuning variables — now calibrated for JERK detection
# Jerk amplifies optical-flow noise, so sensitivity is much higher
# than the old reversal-based threshold.
sensitivity  = .2    # jerk magnitude threshold per axis
qopts        = 0.01    # quality of points selected (lower = more points)
sensitivity2 = 0.7    # ratio of points needed to flag a frame
spread       = 5     # minimum distance between points when initialized
minpts       = 100     # min tracked points before reset (raised for grid)
fb_error_threshold = .5   # max pixels of forward-backward round-trip error
grid_cols=16
grid_rows=12
pts_per_cell=1

st_params = dict(maxCorners=50, qualityLevel=qopts,
                 minDistance=int(spread), blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                           10, 0.03))

color = np.random.randint(0, 255, (200, 3))

cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', TARGET_WIDTH, TARGET_HEIGHT)

# Initial frame setups
old_frame, old_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows, pts_per_cell=pts_per_cell, **st_params)

# State needed for jerk: previous velocity AND previous acceleration per point
old_vec   = [[0, 0]] * (len(p0) if p0 is not None else 0)
old_accel = [[0, 0]] * (len(p0) if p0 is not None else 0)
warmup    = 2   # number of frames before jerk becomes meaningful

new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
    
while True:
    good_old, good_new = track_with_fb_validation(old_gray, new_gray, p0, lk_params, fb_error_threshold)

    new_vec   = []
    new_accel = []

    if good_new is not None and len(good_new) > 0:
        counth = 0
        countv = 0

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            vx, vy = a - c, b - d
            new_vec.append([vx, vy])

            ov = old_vec[i]   if i < len(old_vec)   else [0, 0]
            oa = old_accel[i] if i < len(old_accel) else [0, 0]

            ax = vx - ov[0]
            ay = vy - ov[1]
            new_accel.append([ax, ay])

            jx = ax - oa[0]
            jy = ay - oa[1]

            if warmup == 0:
                if (ax < 0 and oa[0]>0 or ax > 0 and oa[0]<0) and abs(jx) >= sensitivity: counth += 1
                if (ay < 0 and oa[1]>0 or ay > 0 and oa[1]<0) and abs(jy) >= sensitivity: countv += 1

            new_frame = cv.circle(new_frame, (int(a), int(b)), 5,
                                color[i % len(color)].tolist(), -1)

        if warmup == 0 and len(good_new) > 0 and (
            counth > sensitivity2 * len(good_new) or
            countv > sensitivity2 * len(good_new)):
            print('shake?')

    # Draw grid overlay
    for c in range(1, grid_cols):
        x = c * (TARGET_WIDTH // grid_cols)
        cv.line(new_frame, (x, 0), (x, TARGET_HEIGHT), (50, 50, 50), 1)
    for r in range(1, grid_rows):
        y = r * (TARGET_HEIGHT // grid_rows)
        cv.line(new_frame, (0, y), (TARGET_WIDTH, y), (50, 50, 50), 1)

    cv.imshow('frame', new_frame)

    k = cv.waitKey(15) & 0xff
    if k == 27 or k == ord('q'):
        break

    # Shift state forward
    if good_new is not None and len(good_new) > 0:
        old_gray, p0 = shift_data(new_gray, good_new)
        old_vec   = new_vec
        old_accel = new_accel
        if warmup > 0:
            warmup -= 1
    else:
        old_gray = new_gray

    # Read next frame
    new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)

    # Reset if user pressed 'r' or we lost too many points
    if k == ord('r') or p0 is None or len(p0) < minpts:
        p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows,
                                      pts_per_cell=pts_per_cell, **st_params)
        old_vec   = [[0, 0]] * (len(p0) if p0 is not None else 0)
        old_accel = [[0, 0]] * (len(p0) if p0 is not None else 0)
        warmup    = 2
        print('reset')

cap.release()
cv.destroyAllWindows()