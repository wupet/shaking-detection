import numpy as np
import cv2 as cv
import os
import random
import csv  # Added for spreadsheet logging
from functions import shift_data


# Global memory cache to hold downsampled video data arrays
VIDEO_CACHE = {}

def pre_load_dataset():
    """
    Decodes all 24 videos from disk exactly ONCE at startup.
    Converts to grayscale and downsizes to eliminate CPU bottlenecks.
    """
    shaking_dir = "shaking"
    not_shaking_dir = "not shaking"
    
    raw_dataset = []
    for i in range(1, 13):
        raw_dataset.append((os.path.join(shaking_dir, f"shaking{i}.mp4"), True))       
        raw_dataset.append((os.path.join(not_shaking_dir, f"notshaking{i}.mp4"), False))
        
    print("=== STARTING RAM CACHE PRE-LOAD ===")
    print("Reading and downsampling video structures... please wait.")
    
    TARGET_WIDTH = 320  # Optimized size: drops mathematical complexity by 4x
    
    for video_path, is_shaking in raw_dataset:
        if not os.path.exists(video_path):
            print(f"  -> Warning: Skipping missing file: {video_path}")
            continue
            
        cap = cv.VideoCapture(video_path)
        orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Scale handling
        if TARGET_WIDTH < orig_width:
            scale_factor = TARGET_WIDTH / orig_width
            target_height = int(orig_height * scale_factor)
            target_width = TARGET_WIDTH
        else:
            target_height, target_width = orig_height, orig_width
            
        frames_list = []
        while True:
            # Bypass your local exit()-heavy get_proc_next_frame for startup pre-loading
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Immediately downscale to preserve RAM limits
            scaled_gray = cv.resize(gray, (target_width, target_height))
            frames_list.append(scaled_gray)
            
        cap.release()
        VIDEO_CACHE[video_path] = frames_list
        print(f"  -> Cached: {video_path} ({len(frames_list)} frames loaded)")
        
    print(f"\nSuccessfully cached {len(VIDEO_CACHE)} videos directly into system memory!\n")
    return raw_dataset

def check_cached_video_for_shake(video_path, sensitivity, sensitivity2, qopts, spread, minpts):
    """
    Runs your shake extraction math purely inside RAM arrays.
    """
    frames = VIDEO_CACHE.get(video_path, [])
    if not frames or len(frames) < 2:
        return False

    st_params = dict(maxCorners=50, 
                     qualityLevel=qopts, 
                     minDistance=int(spread) if int(spread) > 0 else 1, 
                     blockSize=7)
                     
    lk_params = dict(winSize=(15, 15), 
                     maxLevel=2, 
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # --- Frame 0 & 1 Initial Setup ---
    old_gray = frames[0]
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
    
    new_gray = frames[1]
    if p0 is None:
        return False
        
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
    
    # Fast inline replacement for processpts to avoid drawing circles onto cached arrays
    old_vec = []
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            old_vec.append([a - c, b - d])
        old_gray, p0 = shift_data(new_gray, good_new)
    else:
        p0 = None

    shaking_detected = False

    # --- Process Remainder of Cached Frame Array ---
    for frame_idx in range(2, len(frames)):
        new_gray = frames[frame_idx]

        if p0 is None or len(p0) < int(minpts):
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
            if p0 is None:
                continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        new_vec = []
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            good_new, good_old = [], []

        counth = 0
        countv = 0
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            new_vec.append([a - c, b - d])
            
            if i >= len(old_vec):
                continue

            # Your directional vector reversal check formulas
            if ((new_vec[i][0] < 0 and old_vec[i][0] > 0 and old_vec[i][0] - new_vec[i][0] > sensitivity) or 
                (new_vec[i][0] > 0 and old_vec[i][0] < 0 and new_vec[i][0] - old_vec[i][0] > sensitivity)):
                counth += 1
            if ((new_vec[i][1] < 0 and old_vec[i][1] > 0 and old_vec[i][1] - new_vec[i][1] > sensitivity) or 
                (new_vec[i][1] > 0 and old_vec[i][1] < 0 and new_vec[i][1] - old_vec[i][1] > sensitivity)):
                countv += 1

        if p1 is not None and len(p1) > 0:
            if counth > sensitivity2 * len(p1) or countv > sensitivity2 * len(p1):
                shaking_detected = True
                break  # Shake found! Exit video instantly to save time

        if len(good_new) > 0:
            old_gray, p0 = shift_data(new_gray, good_new)
        else:
            p0 = None
        old_vec = new_vec

    return shaking_detected

def run_random_search():
    # Pre-populate our database cache structure
    dataset = pre_load_dataset()
    
    best_score = -1
    best_params = {}
    
    csv_filename = "tuning_trials.csv"
    
    print("=== STARTING RANDOM SEARCH OPTIMIZATION (100 CYCLES) ===")
    
    # Open CSV file and write header
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Trial", "Score_Out_Of_24", "Sensitivity", "Sensitivity2", "QOpts", "Spread", "MinPts"])

    
        for run in range(1, 501):
            # Sample random hyperparameters based on your requirements
            sensitivity = random.uniform(.001, .4)
            sensitivity2 = random.uniform(0.4, 0.8)
            qopts = random.uniform(0.15, 0.4)
            spread = random.randint(1, 25)
            minpts = random.randint(5, 20)
            
            correct_predictions = 0
            for video_path, expected_is_shaking in dataset:
                detected = check_cached_video_for_shake(video_path, sensitivity, sensitivity2, qopts, spread, minpts)
                if detected == expected_is_shaking:
                    correct_predictions += 1
                    
            print(f"Iteration {run:03d}/500 | Target Accuracy: {correct_predictions:02d}/24 | Current High Record: {max(best_score, correct_predictions)}/24 {sensitivity} {sensitivity2} {qopts} {spread} {minpts}")
            
            # Instantly write this trial's data to the CSV row
            csv_writer.writerow([run, correct_predictions, sensitivity, sensitivity2, qopts, spread, minpts])
            csv_file.flush() # Force write immediately to disk
            
            if correct_predictions > best_score:
                best_score = correct_predictions
                best_params = {
                    "sensitivity": sensitivity,
                    "sensitivity2": sensitivity2,
                    "qopts": qopts,
                    "spread": spread,
                    "minpts": minpts
                }
                
            if best_score == len(dataset):
                print(f"\nOptimization target met! Perfect values matched on iteration {run}!")
                break

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED")
    
    print(f"Data saved successfully to logs: '{csv_filename}'")

    
    print(f"Top Detection Accuracy: {best_score}/{len(dataset)} videos tracked correctly.")
    print("Optimal Parameter Profile Map:")
    for param, val in best_params.items():
        if isinstance(val, float):
            print(f"  -> {param}: {val:.4f}")
        else:
            print(f"  -> {param}: {val}")
    print("="*50)

if __name__ == "__main__":
    run_random_search()
    
# best so far values: 21 correct 0.224761853	0.495295564	0.200998043	9	15
