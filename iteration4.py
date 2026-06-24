import numpy as np
import cv2 as cv
from collections import deque
from functions import *
from dotenv import load_dotenv
import os

def main():
    # ---------------- Sensitivity Variables ----------------
    sensitivity         = 0.15  # jerk magnitude threshold (pixels/frame/frame/frame)
    sensitivity2        = 0.7   # fraction of points that must agree with global accel

    # ---------------- Shi-Tomasi Variables ----------------
    qopts               = 0.01  # minimum quality of points found by ST corner detection
    spread              = 5     # minimum initial distance between points found by ST corner detection
    grid_cols           = 12    # number of columns in the point grid
    grid_rows           = 9     # number of rows in the point grid
    pts_per_cell        = 1     # number of points per cell in the point grid

    # ---------------- Data Quality Varibles ----------------
    fb_error_threshold  = 0.5   # maximum error between forward-backward tracking before throwing point
    minpts              = 30    # minimum number of points before rerunning ST corner detection

    # ---------------- Oscillation Window Variables ----------------
    osc_window          = 10    # how many recent jerk values to store and compare to
    osc_flips_required  = 3     # sign reversals of global jerk inside window

    # ---------------- OpenCV Algorithm Parameters ----------------
    st_params = dict(maxCorners=50, qualityLevel=qopts,
                    minDistance=int(spread), blockSize=7)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                            10, 0.03))

    # ---------------- Debug Variables ---------------- (points, the word shake on screen, lines for grid)
    debug = True                # turns on extra visuals and debug information in terminal and stream
    show_points = True          # toggles whether or not the chosen ST points are displayed
    point_color = (0,0,255)     # the color of the ST points if being displayed (given in BGR)
    point_size = 5              # the size of the ST points if being displayed (in pixels)
    show_grid = True            # toggles whether or not the point grid is displayed
    grid_color = (0,0,0)        # the color of the grid lines if being displayed (given in BGR)
    grid_thickness = 1          # the thickness of the grid lines if being displayed (in pixels)
    print_reset = True          # toggles whether or not reset is printed into the terminal whenever ST points are recalculated
    print_shake = True          # toggles whether or not shake is printed into the window whenever shaking is detected

    load_dotenv()

    # cap = cv.VideoCapture(0)
    # cap = cv.VideoCapture("shaking\\shaking16.mp4")
    cap = cv.VideoCapture(os.getenv("rtsp_url"))

    # calculates new dimensions and scales display to such
    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    TARGET_WIDTH = 640
    if TARGET_WIDTH < orig_width:
        scale_factor = TARGET_WIDTH / orig_width
        TARGET_HEIGHT = int(orig_height * scale_factor)
    else:
        TARGET_HEIGHT = orig_height
        TARGET_WIDTH = orig_width
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', TARGET_WIDTH, TARGET_HEIGHT)

    # ---------------- Initial frame ----------------
    _, old_gray = get_next_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)

    # generates the list of points to track (float32 (N,1,2))
    p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows,
                                pts_per_cell=pts_per_cell, **st_params)

    # Per-point state, kept in lockstep with p0 via the FB good-mask
    old_vec   = np.zeros((len(p0) if p0 is not None else 0, 2), dtype=np.float32)
    old_accel = np.zeros((len(p0) if p0 is not None else 0, 2), dtype=np.float32)

    # Global-motion state (one vector for the whole frame)
    old_global_accel = np.zeros(2, dtype=np.float32)

    # Warmup: need 3 frames before jerk is meaningful (v, then a, then j)
    warmup = 3

    # History of recent global jerk vectors for oscillation detection
    jerk_history = deque(maxlen=osc_window)

    new_frame, new_gray = get_next_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
    frame_idx = 2
    reset_ct = 1
    
    while new_frame is not None:
        good_old, good_new, good_mask = track_with_fb_validation(
            old_gray, new_gray, p0, lk_params, fb_error_threshold)

        shake_detected = False

        if good_new is not None and len(good_new) > 0:
            # filters out old vectors, accelerations that were not tracked
            old_vec   = old_vec[good_mask]
            old_accel = old_accel[good_mask]

            # Per-point velocity and acceleration
            new_vec   = (good_new.reshape(-1, 2) - good_old.reshape(-1, 2)).astype(np.float32)
            new_accel = new_vec - old_vec

            # --- Global motion ---
            # Median is robust against a minority of points on a shaking object.
            global_a = np.median(new_accel, axis=0)
            global_j = global_a - old_global_accel

            # --- Agreement: fraction of points whose acceleration sign
            #     matches the global acceleration on each axis ---
            if abs(global_a[0]) > 1e-6:
                agree_x = np.mean(np.sign(new_accel[:, 0]) == np.sign(global_a[0]))
            else:
                agree_x = 0.0
            if abs(global_a[1]) > 1e-6:
                agree_y = np.mean(np.sign(new_accel[:, 1]) == np.sign(global_a[1]))
            else:
                agree_y = 0.0

            # --- Record global jerk for oscillation analysis ---
            if warmup == 0:
                jerk_history.append(global_j.copy())

                # Count sign flips of global jerk per axis across the window.
                # Only flips whose magnitudes exceed sensitivity count, so
                # tiny noise wiggles are ignored.
                if len(jerk_history) >= 3:
                    jh = np.array(jerk_history)
                    sig_x = np.sign(np.where(np.abs(jh[:, 0]) >= sensitivity, jh[:, 0], 0))
                    sig_y = np.sign(np.where(np.abs(jh[:, 1]) >= sensitivity, jh[:, 1], 0))

                    flips_x = np.sum((sig_x[1:] != 0) & (sig_x[:-1] != 0)
                                    & (sig_x[1:] != sig_x[:-1]))
                    flips_y = np.sum((sig_y[1:] != 0) & (sig_y[:-1] != 0)
                                    & (sig_y[1:] != sig_y[:-1]))

                    strong_x = (abs(global_j[0]) >= sensitivity
                                and agree_x >= sensitivity2
                                and flips_x >= osc_flips_required)
                    strong_y = (abs(global_j[1]) >= sensitivity
                                and agree_y >= sensitivity2
                                and flips_y >= osc_flips_required)

                    if strong_x or strong_y:
                        shake_detected = True

            if debug:
                if show_points:
                    # --- Draw points (stable colors via index in surviving array) ---
                    for i, pt in enumerate(good_new):
                        a, b = pt.ravel()
                        new_frame = cv.circle(new_frame, (int(a), int(b)), point_size, point_color, -1)
                if show_grid:
                    # Draw grid overlay
                    for c in range(1, grid_cols):
                        x = c * (TARGET_WIDTH // grid_cols)
                        cv.line(new_frame, (x, 0), (x, TARGET_HEIGHT), grid_color, grid_thickness)
                    for r in range(1, grid_rows):
                        y = r * (TARGET_HEIGHT // grid_rows)
                        cv.line(new_frame, (0, y), (TARGET_WIDTH, y), grid_color, grid_thickness)
                if shake_detected and print_shake:
                    cv.putText(new_frame, 'SHAKE', (20, 40),cv.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
                if warmup > 0 and print_reset:
                    cv.putText(new_frame, 'RESET', (140, 40),cv.FONT_HERSHEY_PLAIN, 2.0, (157, 0, 255), 2)

            if shake_detected:
                print(f'frame: {frame_idx}; SHAKE')

            # --- Shift state forward ---
            old_gray         = new_gray 
            p0               = good_new
            old_vec          = new_vec
            old_accel        = new_accel
            old_global_accel = global_a

            if warmup > 0:
                warmup -= 1
                
        else:
            # Tracking failed this frame; just advance the gray reference.
            old_gray = new_gray
            p0 = None  # force reset below

        cv.imshow('frame', new_frame)
        k = cv.waitKey(1) & 0xff
        if k == 27 or k == ord('q'):
            break

        # Reset if user pressed 'r' or we lost too many points
        if k == ord('r') or p0 is None or len(p0) < minpts:
            p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows,
                                        pts_per_cell=pts_per_cell, **st_params)
            n = len(p0) if p0 is not None else 0
            old_vec          = np.zeros((n, 2), dtype=np.float32)
            old_accel        = np.zeros((n, 2), dtype=np.float32)
            old_global_accel = np.zeros(2, dtype=np.float32)
            jerk_history.clear()
            warmup = 3
            reset_ct+=1
            if debug and print_reset:
                print(f'reset: {reset_ct}')

        # Advance to next frame
        new_frame, new_gray = get_next_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
        frame_idx += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()