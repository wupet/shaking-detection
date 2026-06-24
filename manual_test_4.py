"""
Batch tester for the shake detector.

Folder layout expected:
    shaking/
        shaking1.mp4
        shaking2.mp4
        ...
    notshaking/
        notshaking1.mp4
        notshaking2.mp4
        ...

For each video the detector runs headlessly. A video is classified as
"shake detected" if the detector fires on at least `min_trigger_frames`
frames (default 1). Adjust that threshold if you want to require
sustained detection instead of a single hit.
"""

import os
import glob
import time
import numpy as np
import cv2 as cv
from collections import deque

from functions import (grid_goodFeaturesToTrack, track_with_fb_validation, get_next_frame)

# ---------------- Detector tuning (mirror iteration3.py) ----------------
sensitivity         = 0.15
qopts               = 0.01
sensitivity2        = 0.7
spread              = 5
minpts              = 30
fb_error_threshold  = 0.5

grid_cols           = 12
grid_rows           = 9
pts_per_cell        = 1

osc_window          = 10
osc_flips_required  = 3

TARGET_WIDTH        = 640    # match iteration3.py
min_trigger_frames  = 1      # frames of detection needed to label a video "shake"

st_params = dict(maxCorners=50, qualityLevel=qopts,
                 minDistance=int(spread), blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                           10, 0.03))


def run_detector_on_video(path, verbose=False):
    """
    Returns dict with:
      detected (bool), trigger_frames (int), total_frames (int),
      resets (int), elapsed (float seconds)
    """
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        return None

    ow = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if ow == 0 or oh == 0:
        cap.release()
        return None

    if TARGET_WIDTH < ow:
        th = int(oh * (TARGET_WIDTH / ow))
        tw = TARGET_WIDTH
    else:
        tw, th = ow, oh

    _, old_gray = get_next_frame(cap, tw, th)
    if old_gray is None:
        cap.release()
        return None

    p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows,
                                  pts_per_cell=pts_per_cell, **st_params)

    n0 = len(p0) if p0 is not None else 0
    old_vec          = np.zeros((n0, 2), dtype=np.float32)
    old_accel        = np.zeros((n0, 2), dtype=np.float32)
    old_global_accel = np.zeros(2, dtype=np.float32)
    warmup           = 3
    jerk_history     = deque(maxlen=osc_window)

    trigger_frames = 0
    total_frames   = 0
    resets         = 0
    t0 = time.time()

    new_frame, new_gray = get_next_frame(cap, tw, th)
    while new_frame is not None:
        total_frames += 1

        # NOTE: requires the patched 3-return-value version of
        # track_with_fb_validation. If your functions.py still returns
        # only 2 values, replace the next line accordingly.
        result = track_with_fb_validation(old_gray, new_gray, p0,
                                          lk_params, fb_error_threshold)
        if len(result) == 3:
            good_old, good_new, good_mask = result
        else:
            good_old, good_new = result
            good_mask = None

        shake = False

        if good_new is not None and len(good_new) > 0:
            if good_mask is not None:
                old_vec   = old_vec[good_mask]
                old_accel = old_accel[good_mask]

            new_vec   = (good_new.reshape(-1, 2)
                         - good_old.reshape(-1, 2)).astype(np.float32)

            # Guard against shape mismatch if running with the old
            # 2-return-value tracker.
            if new_vec.shape[0] != old_vec.shape[0]:
                old_vec   = np.zeros_like(new_vec)
                old_accel = np.zeros_like(new_vec)

            new_accel = new_vec - old_vec

            global_a = np.median(new_accel, axis=0)
            global_j = global_a - old_global_accel

            if abs(global_a[0]) > 1e-6:
                agree_x = np.mean(np.sign(new_accel[:, 0]) == np.sign(global_a[0]))
            else:
                agree_x = 0.0
            if abs(global_a[1]) > 1e-6:
                agree_y = np.mean(np.sign(new_accel[:, 1]) == np.sign(global_a[1]))
            else:
                agree_y = 0.0

            if warmup == 0:
                jerk_history.append(global_j.copy())
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
                        shake = True

            if shake:
                trigger_frames += 1

            old_gray         = new_gray 
            p0               = good_new
            old_vec          = new_vec
            old_accel        = new_accel
            old_global_accel = global_a
            if warmup > 0:
                warmup -= 1
        else:
            old_gray = new_gray
            p0 = None

        if p0 is None or len(p0) < minpts:
            p0 = grid_goodFeaturesToTrack(old_gray, cols=grid_cols, rows=grid_rows,
                                          pts_per_cell=pts_per_cell, **st_params)
            n = len(p0) if p0 is not None else 0
            old_vec          = np.zeros((n, 2), dtype=np.float32)
            old_accel        = np.zeros((n, 2), dtype=np.float32)
            old_global_accel = np.zeros(2, dtype=np.float32)
            jerk_history.clear()
            warmup = 3
            resets += 1

        new_frame, new_gray = get_next_frame(cap, tw, th)

    cap.release()
    elapsed = time.time() - t0

    return dict(
        detected       = trigger_frames >= min_trigger_frames,
        trigger_frames = trigger_frames,
        total_frames   = total_frames,
        resets         = resets,
        elapsed        = elapsed,
    )


def enumerate_videos(folder, prefix):
    """Find files named <prefix>1.mp4, <prefix>2.mp4, ... in order."""
    if not os.path.isdir(folder):
        return []
    # Discover all matching files, then sort numerically
    pattern = os.path.join(folder, f"{prefix}*.mp4")
    files = glob.glob(pattern)
    def num_key(p):
        name = os.path.splitext(os.path.basename(p))[0]
        digits = ''.join(ch for ch in name[len(prefix):] if ch.isdigit())
        return int(digits) if digits else 0
    files.sort(key=num_key)
    return files


def evaluate(folder, prefix, expected_label):
    """expected_label: True means shaking, False means not shaking."""
    videos = enumerate_videos(folder, prefix)
    results = []
    print(f"\n=== {folder} ({len(videos)} videos, expected={'SHAKE' if expected_label else 'NO SHAKE'}) ===")
    for path in videos:
        r = run_detector_on_video(path)
        if r is None:
            print(f"  [SKIP] {os.path.basename(path)}  (could not open)")
            continue
        correct = (r['detected'] == expected_label)
        mark = 'OK ' if correct else 'XX '
        print(f"  {mark}{os.path.basename(path):25s} "
              f"detected={str(r['detected']):5s}  "
              f"triggers={r['trigger_frames']:4d}/{r['total_frames']:<4d}  "
              f"resets={r['resets']:<3d}  "
              f"time={r['elapsed']:.2f}s")
        results.append((path, r, correct))
    return results


def main():
    shake_results    = evaluate("shaking",    "shaking",    expected_label=True)
    notshake_results = evaluate("notshaking", "notshaking", expected_label=False)

    # Confusion matrix
    TP = sum(1 for _, r, _ in shake_results    if r['detected'])
    FN = sum(1 for _, r, _ in shake_results    if not r['detected'])
    FP = sum(1 for _, r, _ in notshake_results if r['detected'])
    TN = sum(1 for _, r, _ in notshake_results if not r['detected'])

    total = TP + FN + FP + TN
    if total == 0:
        print("\nNo videos were evaluated.")
        return

    acc  = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    print("\n=== Summary ===")
    print(f"  True  Positives (shake correctly flagged):    {TP}")
    print(f"  False Negatives (shake missed):                {FN}")
    print(f"  False Positives (not-shake wrongly flagged):   {FP}")
    print(f"  True  Negatives (not-shake correctly ignored): {TN}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")


if __name__ == "__main__":
    main()