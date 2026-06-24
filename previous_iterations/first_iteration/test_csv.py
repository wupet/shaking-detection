import numpy as np
import cv2 as cv
import os
import csv
from functions import shift_data
# not working after shifting stuff into first iteration.
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
        raw_dataset.append((os.path.join(shaking_dir, f"../shaking{i}.mp4"), True))       
        raw_dataset.append((os.path.join(not_shaking_dir, f"../notshaking{i}.mp4"), False))
        
    print("=== STARTING RAM CACHE PRE-LOAD ===")
    print("Reading and downsampling video structures... please wait.")
    
    TARGET_WIDTH = 320  
    
    for video_path, is_shaking in raw_dataset:
        if not os.path.exists(video_path):
            print(f"  -> Warning: Skipping missing file: {video_path}")
            continue
            
        cap = cv.VideoCapture(video_path)
        orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        if TARGET_WIDTH < orig_width:
            scale_factor = TARGET_WIDTH / orig_width
            target_height = int(orig_height * scale_factor)
            target_width = TARGET_WIDTH
        else:
            target_height, target_width = orig_height, orig_width
            
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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

    old_gray = frames[0]
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **st_params)
    
    new_gray = frames[1]
    if p0 is None:
        return False
        
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
    
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

            if ((new_vec[i][0] < 0 and old_vec[i][0] > 0 and old_vec[i][0] - new_vec[i][0] > sensitivity) or 
                (new_vec[i][0] > 0 and old_vec[i][0] < 0 and new_vec[i][0] - old_vec[i][0] > sensitivity)):
                counth += 1
            if ((new_vec[i][1] < 0 and old_vec[i][1] > 0 and old_vec[i][1] - new_vec[i][1] > sensitivity) or 
                (new_vec[i][1] > 0 and old_vec[i][1] < 0 and new_vec[i][1] - old_vec[i][1] > sensitivity)):
                countv += 1

        if p1 is not None and len(p1) > 0:
            if counth > sensitivity2 * len(p1) or countv > sensitivity2 * len(p1):
                shaking_detected = True
                break  

        if len(good_new) > 0:
            old_gray, p0 = shift_data(new_gray, good_new)
        else:
            p0 = None
        old_vec = new_vec

    return shaking_detected

def run_csv_validation():
    csv_filename = "manual_trials.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Error: Could not find the CSV log file named '{csv_filename}' in this directory.")
        return

    dataset = pre_load_dataset()
    print("=== STARTING CSV LOG VALIDATION ===")

    with open(csv_filename, mode='r') as csv_file:
        # Use DictReader so we can access variables by their column headers safely
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            trial_id = row["Trial"]
            
            # Cast inputs out of strings back to their proper computational numeric formats
            sensitivity = float(row["Sensitivity"])
            sensitivity2 = float(row["Sensitivity2"])
            qopts = float(row["QOpts"])
            spread = int(row["Spread"])
            minpts = int(row["MinPts"])
            
            correct_predictions = 0
            failed_videos = []
            
            for video_path, expected_is_shaking in dataset:
                detected = check_cached_video_for_shake(video_path, sensitivity, sensitivity2, qopts, spread, minpts)
                
                if detected == expected_is_shaking:
                    correct_predictions += 1
                else:
                    # Log precisely which folder item tripped up this unique parameter combo
                    failed_videos.append(video_path)
            
            # Print performance layout summary output per row element block
            print(f"Trial {int(trial_id):03d} | Re-verified Score: {correct_predictions:02d}/24")
            if failed_videos:
                print(f"  └── ❌ Failed on: {', '.join(failed_videos)}")
            else:
                print("  └──  Perfect Validation Match!")
                
    print("\n=== VALIDATION RUN COMPLETE ===")

if __name__ == "__main__":
    run_csv_validation()