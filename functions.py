import numpy as np
import cv2 as cv

def grid_goodFeaturesToTrack(gray, cols=4, rows=3, pts_per_cell=2, **st_params):
    """
    Splits the image into cols x rows cells and finds `pts_per_cell`
    Shi-Tomasi corners inside each cell. Returns points in the same
    (N, 1, 2) float32 shape as cv.goodFeaturesToTrack.
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
    old_gray = new_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return old_gray, p0


def get_proc_next_frame(cap):
    ret, new_frame = cap.read()
    if not ret:
        # print("Cannot read video file")
        return None, None
    return new_frame, cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)


def get_scaled_frame(cap, target_w, target_h):
    frame, gray = get_proc_next_frame(cap)
    if frame is None:
        return None, None
    scaled_frame = cv.resize(frame, (target_w, target_h))
    scaled_gray  = cv.resize(gray,  (target_w, target_h))
    return scaled_frame, scaled_gray

def track_with_fb_validation(prev_gray, curr_gray, prev_pts, lk_params,
                              fb_error_threshold=1.0):
    """
    Forward-backward Lucas-Kanade tracking with error validation.
    Returns (good_prev_pts, good_next_pts) — both filtered to only
    include points that round-trip within fb_error_threshold pixels.
    Returns (None, None) if tracking fails or too few points survive.
    """
    # Forward: prev -> curr
    next_pts, status_fwd, _ = cv.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params)
    if next_pts is None:
        return None, None

    # Backward: curr -> prev
    back_pts, status_bwd, _ = cv.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, next_pts, None, **lk_params)
    if back_pts is None:
        return None, None

    # Distance between original and round-tripped points
    fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).reshape(-1)

    good = ((status_fwd.reshape(-1) == 1)
            & (status_bwd.reshape(-1) == 1)
            & (fb_error < fb_error_threshold))

    if good.sum() < 1:
        return None, None

    return prev_pts[good], next_pts[good]