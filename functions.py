import numpy as np
import cv2 as cv

def grid_goodFeaturesToTrack(gray, cols=12, rows=9, pts_per_cell=1, **st_params):
    """ Detect Shi-Tomasi corners distributed evenly across a grid over the image.

    The input grayscale image is partitioned into a `cols` x `rows` grid of
    rectangular cells. For each cell, `cv.goodFeaturesToTrack` is run on the
    local region of interest (ROI) with `maxCorners` capped at `pts_per_cell`
    and `minDistance` clamped so it never exceeds a third of the cell's
    smaller dimension (this prevents the parameter from suppressing all
    candidates in small cells). Points returned in cell-local coordinates are
    shifted back to global image coordinates by adding the cell's top-left
    offset. The remaining cells on the right and bottom edges are extended to
    the image border to absorb any pixels lost to integer division. All
    per-cell point arrays are concatenated and returned in the canonical
    (N, 1, 2) float32 shape used by OpenCV; if no cell produced any corners,
    `None` is returned.
   """
    
    h, w = gray.shape[:2]
    cell_w = w // cols
    cell_h = h // rows

    cell_params = dict(st_params)
    cell_params['maxCorners'] = pts_per_cell
    cell_params['minDistance'] = max(1, min(cell_params.get('minDistance', 7),
                                            min(cell_w, cell_h) // 3))

    all_pts = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1 = w if c == cols - 1 else x0 + cell_w
            y1 = h if r == rows - 1 else y0 + cell_h

            roi = gray[y0:y1, x0:x1]
            pts = cv.goodFeaturesToTrack(roi, mask=None, **cell_params)
            if pts is not None:
                pts[:, 0, 0] += x0
                pts[:, 0, 1] += y0
                all_pts.append(pts)

    if not all_pts:
        return None
    return np.concatenate(all_pts, axis=0).astype(np.float32)


def shift_data(new_gray, good_new):
    """ Promote the current frame's state to become the "previous" state for the next iteration.

    In an optical-flow loop the freshly processed grayscale frame and the
    points that survived tracking need to be rolled forward so that, on the
    next iteration, they play the role of "old". This helper makes a copy of
    the current grayscale image (so subsequent in-place operations do not
    corrupt history) and reshapes the surviving points back into the
    (N, 1, 2) layout that `cv.calcOpticalFlowPyrLK` expects as input.
    """
    
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return old_gray, p0


def get_proc_next_frame(cap):
    """ Read the next frame from a video capture and return both color and grayscale versions.

    The function calls `cap.read()` on the supplied `cv.VideoCapture` object.
    If the capture has ended or fails, `(None, None)` is returned so the
    caller can break out of its loop cleanly. Otherwise the BGR frame is
    converted to a single-channel grayscale image (suitable for feature
    detection and Lucas-Kanade tracking) and both versions are returned.
    """
    
    ret, new_frame = cap.read()
    if not ret:
        return None, None
    return new_frame, cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)


def get_scaled_frame(cap, target_w, target_h):
    """ Read the next frame and resize both its color and grayscale versions to a target resolution.

    This is a thin wrapper around `get_proc_next_frame`: it first pulls the
    next BGR frame and its grayscale counterpart from the capture, and if
    the stream is exhausted it propagates `(None, None)`. Both images are
    then resampled with `cv.resize` to the requested `target_w` x `target_h`
    dimensions so downstream processing operates on a known, fixed-size
    canvas regardless of the source resolution.
    """
    
    frame, gray = get_proc_next_frame(cap)
    if frame is None:
        return None, None
    scaled_frame = cv.resize(frame, (target_w, target_h))
    scaled_gray = cv.resize(gray, (target_w, target_h))
    return scaled_frame, scaled_gray


def track_with_fb_validation(prev_gray, curr_gray, prev_pts, lk_params,
                             fb_error_threshold=1.0):
    """ Track points between two frames with forward-backward Lucas-Kanade consistency checking.

    The function performs three steps. First, a forward pyramidal LK pass
    estimates where each point in `prev_pts` lands in `curr_gray`. Second,
    a backward pass tracks those predicted locations from `curr_gray` back
    into `prev_gray`. Third, the round-trip error is computed as the
    Euclidean distance between the original points and their back-tracked
    positions; a point is considered reliable only if both LK passes
    reported success (`status == 1`) and the round-trip error is below
    `fb_error_threshold`. The function returns a tuple
    `(good_prev_pts, good_next_pts, good_mask)` where `good_mask` is a
    boolean array aligned with the original `prev_pts`. Callers can use
    this mask to filter parallel state arrays (e.g. velocity, acceleration)
    in lockstep with the surviving points. If the input is empty, either
    LK call fails, or no point passes the consistency check,
    `(None, None, None)` is returned.
    """
    
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    # Forward: prev -> curr
    next_pts, status_fwd, _ = cv.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params)
    if next_pts is None:
        return None, None, None

    # Backward: curr -> prev
    back_pts, status_bwd, _ = cv.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_pts, None, **lk_params)
    if back_pts is None:
        return None, None, None

    # Distance between original and round-tripped points
    fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).reshape(-1)

    good = ((status_fwd.reshape(-1) == 1)
            & (status_bwd.reshape(-1) == 1)
            & (fb_error < fb_error_threshold))

    if good.sum() < 1:
        return None, None, None

    return prev_pts[good], next_pts[good], good