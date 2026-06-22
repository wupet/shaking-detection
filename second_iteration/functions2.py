import cv2
import numpy as np
from second_iteration.functions2 import *
from collections import deque
from scipy.signal import butter, filtfilt

class CameraShakeDetector:
    """provides a way to detect shaking in a video processed by opencv 
    
    This class takes in various tuning variables and uses them to calibrate 
    various functions used to detect shaking. We will be using 
        - grid-based Shi-Tomasi point selection to find points to track
        - Lucas-Kanade point tracking to measure movement
        - forward-backward error checks to check point validity after LK
        - Butterworth high-pass digital filters to remove smooth movement 
        - Directional flip counters to separate pans from shakes
        - opencv's estimateAffinepartial2d function to interpret movement 
    """
    
    def __init__(               # values passed into constructor and their defaults
        self,
        target_height=240,      # height we reduce videos to (240p)
        grid_rows=6,            # number of rows used when distributing points
        grid_cols=8,            # number of cols used when distributing points
        features_per_cell=4,    # number of Shi-Tomasi corner points per cell
        redetect_interval=15,   # number frames between each redetection cyce
        min_tracks_ratio=0.5,   # minimum ratio of points tracked to maximum points tracked before redetection 
        fb_error_threshold=1.0, # how much error we expect to see in our Lucas-Kanade tracking before removing a point
        motion_buffer_size=30,  # size of the buffer we use to be able to track velocities over time to 
        highpass_window=15,     # length of recent window evaluated for tracking metrics
        shake_threshold=0.001,  # fraction of frame width (RMS)
        inlier_ratio_threshold=0.9 # MINIMUM % of points that must agree on global motion to trust it
    ):
        self.target_height = target_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.features_per_cell = features_per_cell
        self.redetect_interval = redetect_interval
        self.min_tracks_ratio = min_tracks_ratio
        self.fb_error_threshold = fb_error_threshold
        self.shake_threshold = shake_threshold
        self.inlier_ratio_threshold = inlier_ratio_threshold

        # Motion signal buffers (per-frame dx, dy, dtheta)
        self.motion_buffer = deque(maxlen=motion_buffer_size)
        self.highpass_window = highpass_window

        # State
        self.prev_gray = None
        self.prev_points = None
        self.initial_point_count = 0
        self.frame_idx = 0
        self.scale = 1.0

        # LK params
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def _preprocess(self, frame):
        """Returns a shrunk and grayscaled frame """                                                                     
        h, w = frame.shape[:2]
        self.scale = self.target_height / h
        new_w = int(w * self.scale)
        small = cv2.resize(frame, (new_w, self.target_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def _detect_features_grid(self, gray):
        """Detect good features within each grid cell for spatial distribution."""
        h, w = gray.shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        all_points = []

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                cell = gray[y0:y1, x0:x1]

                pts = cv2.goodFeaturesToTrack(
                    cell,
                    maxCorners=self.features_per_cell,
                    qualityLevel=0.01,
                    minDistance=5,
                    blockSize=3,
                )
                if pts is not None:
                    pts[:, 0, 0] += x0
                    pts[:, 0, 1] += y0
                    all_points.append(pts)

        if not all_points:
            return None
        return np.vstack(all_points).astype(np.float32)

    def _track_with_validation(self, prev_gray, curr_gray, prev_pts):
        """Forward-backward LK tracking with error validation."""
        # Forward
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        if next_pts is None:
            return None, None

        # Backward
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, next_pts, None, **self.lk_params
        )
        if back_pts is None:
            return None, None

        # Forward-backward error
        fb_error = np.linalg.norm(prev_pts - back_pts, axis=2).reshape(-1)
        good = (
            (status_fwd.reshape(-1) == 1)
            & (status_bwd.reshape(-1) == 1)
            & (fb_error < self.fb_error_threshold)
        )

        if good.sum() < 6:  # need at least a few for affine estimate
            return None, None

        return prev_pts[good], next_pts[good]

    def _highpass_motion(self):
        """Applies a tight digital Butterworth High-Pass Filter to eliminate smooth panning."""
        if len(self.motion_buffer) < 15:
            return None
            
        arr = np.array(self.motion_buffer)  # shape (N, 3): dx, dy, dtheta
        
        # Keep Wn at 0.05 to ensure slow panning is stripped out mathematically
        b, a = butter(N=2, Wn=0.05, btype='high', analog=False)
        
        try:
            high_dx = filtfilt(b, a, arr[:, 0])
            high_dy = filtfilt(b, a, arr[:, 1])
            high_dtheta = filtfilt(b, a, arr[:, 2])
            
            high = np.stack([high_dx, high_dy, high_dtheta], axis=1)
            return high
        except ValueError:
            return None

    def _compute_shake_score(self, high):
        """Computes shake score using a strict volatility ratio to aggressively lock out panning."""
        recent = high[-self.highpass_window :]
        n_frames = len(recent)
        
        # 1. Base High-Passed Displacement Magnitude
        trans_mag = np.sqrt(recent[:, 0] ** 2 + recent[:, 1] ** 2)
        rms_trans = np.sqrt(np.mean(trans_mag ** 2))
        
        # 2. Track Volatility (Directional Flips Ratio)
        dx_signs = np.sign(recent[:, 0])
        dy_signs = np.sign(recent[:, 1])
        
        # Count actual direction flips
        dx_flips = np.sum(np.diff(dx_signs) != 0)
        dy_flips = np.sum(np.diff(dy_signs) != 0)
        max_possible_flips = (n_frames - 1) * 2
        
        # Calculate what percentage of frames actively vibrated back-and-forth
        flip_ratio = (dx_flips + dy_flips) / max_possible_flips if max_possible_flips > 0 else 0.0
        
        # 3. Explicit Panning Lock
        # Check raw cumulative direction over the buffer window.
        # If the camera travelled a net distance consistently, it's a pan, NOT a shake.
        net_dx = np.sum(recent[:, 0])
        net_dy = np.sum(recent[:, 1])
        abs_sum_dx = np.sum(np.abs(recent[:, 0]))
        abs_sum_dy = np.sum(np.abs(recent[:, 1]))
        
        # A ratio near 1.0 means movement was entirely one-directional (smooth pan)
        pan_consistency_x = abs(net_dx) / abs_sum_dx if abs_sum_dx > 0 else 0
        pan_consistency_y = abs(net_dy) / abs_sum_dy if abs_sum_dy > 0 else 0
        is_panning_consistently = (pan_consistency_x > 0.6) or (pan_consistency_y > 0.6)

        # 4. Compute Acceleration 
        acceleration = np.diff(recent, axis=0) 
        accel_mag = np.sqrt(acceleration[:, 0]**2 + acceleration[:, 1]**2)
        rms_accel = np.sqrt(np.mean(accel_mag ** 2)) if len(accel_mag) > 0 else 0.0
        
        frame_w = self.prev_gray.shape[1] if self.prev_gray is not None else 1
        
        # HARD CRUSH MULTIPLIERS:
        # If it moves too consistently in one direction, or if the direction flip ratio
        # is low (less than 25% of frames flipped), it is completely muted.
        if is_panning_consistently or flip_ratio < 0.25:
            modifier = 0.01  # Crushes smooth pans, pan starts, and pan stops to 0
        else:
            modifier = 2.0   # Amplifies multi-directional micro-vibrations
            
        total_shake_score = ((rms_trans / frame_w) + (rms_accel / frame_w) * 1.5) * modifier
        rms_rot = np.sqrt(np.mean(recent[:, 2] ** 2))

        return total_shake_score, rms_rot

    def process_frame(self, frame):
        """Process one frame. Returns dict with motion + shake info."""
        gray = self._preprocess(frame)
        result = {
            "frame_idx": self.frame_idx,
            "dx": 0.0,
            "dy": 0.0,
            "dtheta": 0.0,
            "n_tracks": 0,
            "inlier_ratio": 0.0,
            "shake_score": 0.0,
            "rot_score": 0.0,
            "is_shaking": False,
        }

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = self._detect_features_grid(gray)
            self.initial_point_count = (
                len(self.prev_points) if self.prev_points is not None else 0
            )
            self.frame_idx += 1
            return result

        need_redetect = (
            self.prev_points is None
            or self.frame_idx % self.redetect_interval == 0
            or len(self.prev_points)
            < self.min_tracks_ratio * max(self.initial_point_count, 1)
        )
        if need_redetect:
            self.prev_points = self._detect_features_grid(self.prev_gray)
            if self.prev_points is not None:
                self.initial_point_count = max(
                    self.initial_point_count, len(self.prev_points)
                )

        if self.prev_points is None or len(self.prev_points) < 6:
            self.prev_gray = gray
            self.frame_idx += 1
            return result

        good_prev, good_next = self._track_with_validation(
            self.prev_gray, gray, self.prev_points
        )

        if good_prev is None:
            self.prev_gray = gray
            self.prev_points = None
            self.frame_idx += 1
            return result

        M, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )

        if M is not None and inliers is not None:
            inlier_ratio = np.sum(inliers) / len(inliers) if len(inliers) > 0 else 0.0
            result["inlier_ratio"] = inlier_ratio

            # Smooth panning generates uniform, high-consensus matrices.
            # We preserve this data to feed it directly to our direction filters.
            if inlier_ratio < self.inlier_ratio_threshold:
                dx, dy, dtheta = 0.0, 0.0, 0.0
            else:
                dx = M[0, 2]
                dy = M[1, 2]
                dtheta = np.arctan2(M[1, 0], M[0, 0])

            self.motion_buffer.append([dx, dy, dtheta])

            result["dx"] = dx
            result["dy"] = dy
            result["dtheta"] = dtheta
            result["n_tracks"] = len(good_next)

            high = self._highpass_motion()
            if high is not None:
                total_shake_score, rms_rot = self._compute_shake_score(high)
                result["shake_score"] = total_shake_score
                result["rot_score"] = rms_rot
                result["is_shaking"] = total_shake_score > self.shake_threshold

            inlier_mask = inliers.reshape(-1).astype(bool)
            self.prev_points = good_next[inlier_mask].reshape(-1, 1, 2)
        else:
            self.prev_points = good_next.reshape(-1, 1, 2)
            self.motion_buffer.append([0.0, 0.0, 0.0])

        self.prev_gray = gray
        self.frame_idx += 1
        return result