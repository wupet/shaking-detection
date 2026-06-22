import numpy as np
import cv2 as cv
from functions import (grid_goodFeaturesToTrack, shift_data,
                       get_scaled_frame, track_with_fb_validation)

def detect_shake_in_video(video_path,
                          sensitivity=0.2,
                          qopts=0.01,
                          sensitivity2=0.9,
                          spread=5,
                          minpts=20,
                          fb_error_threshold=0.5,
                          shake_frame_ratio=0.15,
                          frame_skip=1,
                          grid_cols=4,
                          grid_rows=3,
                          pts_per_cell=4):
    """
    Headless jerk-based shake detector with forward-backward LK validation.
    Returns (is_shaking: bool, shake_ratio: float, frames_processed: int).
    """
    st_params = dict(maxCorners=50, qualityLevel=qopts,
                     minDistance=int(spread), blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                               10, 0.03))

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, 0

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    TARGET_WIDTH = 640
    if TARGET_WIDTH < orig_width:
        scale_factor = TARGET_WIDTH / orig_width
        TARGET_HEIGHT = int(orig_height * scale_factor)
    else:
        TARGET_HEIGHT = orig_height
        TARGET_WIDTH = orig_width

    old_frame, old_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
    if old_gray is None:
        cap.release()
        return None, 0.0, 0

    p0 = grid_goodFeaturesToTrack(old_gray, grid_cols, grid_rows,
                                  pts_per_cell, **st_params)
    old_vec   = [[0, 0]] * (len(p0) if p0 is not None else 0)
    old_accel = [[0, 0]] * (len(p0) if p0 is not None else 0)
    warmup    = 2

    shake_frames = 0
    total_frames = 0

    while True:
        # Frame skipping: grab() decodes header only (cheap)
        for _ in range(frame_skip - 1):
            if not cap.grab():
                break

        new_frame, new_gray = get_scaled_frame(cap, TARGET_WIDTH, TARGET_HEIGHT)
        if new_gray is None:
            break

        # Reset if we have no points or too few
        if p0 is None or len(p0) < minpts:
            p0 = grid_goodFeaturesToTrack(old_gray, grid_cols, grid_rows,
                                          pts_per_cell, **st_params)
            old_gray  = new_gray
            old_vec   = [[0, 0]] * (len(p0) if p0 is not None else 0)
            old_accel = [[0, 0]] * (len(p0) if p0 is not None else 0)
            warmup    = 2
            continue

        # Forward-backward validated LK tracking
        good_old, good_new = track_with_fb_validation(
            old_gray, new_gray, p0, lk_params, fb_error_threshold)

        total_frames += 1

        if good_new is not None and len(good_new) > 0:
            counth = countv = 0
            new_vec   = []
            new_accel = []

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
                    if abs(jx) > sensitivity: counth += 1
                    if abs(jy) > sensitivity: countv += 1

            if warmup == 0 and len(good_new) > 0 and (
                counth > sensitivity2 * len(good_new) or
                countv > sensitivity2 * len(good_new)):
                shake_frames += 1

            old_gray, p0 = shift_data(new_gray, good_new)
            old_vec   = new_vec
            old_accel = new_accel
            if warmup > 0:
                warmup -= 1
        else:
            # FB validation failed — force a reset next loop
            old_gray = new_gray
            p0 = None

    cap.release()
    ratio = shake_frames / total_frames if total_frames > 0 else 0.0
    return ratio > shake_frame_ratio, ratio, total_frames