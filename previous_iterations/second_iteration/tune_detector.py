import os
import cv2
import itertools
import numpy as np
from previous_iterations.second_iteration.functions2 import CameraShakeDetector

# Exact Folder and Prefix Mapping
CATEGORIES = {
    "shaking": {"folder": "shaking", "prefix": "shaking", "expected": True},
    "not shaking": {"folder": "notshaking", "prefix": "notshaking", "expected": False}
}

def get_video_list():
    """Gathers all existing video paths and their expected ground truths."""
    video_list = []
    for cat_name, config in CATEGORIES.items():
        for i in range(1, 13):
            filename = f"{config['prefix']}{i}.mp4"
            video_path = os.path.join(config["folder"], filename)
            if os.path.exists(video_path):
                video_list.append((video_path, config["expected"], filename))
    return video_list

def load_single_video_to_ram(video_path):
    """Opens a video file and reads all frames directly into a list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def run_optimization():
    video_list = get_video_list()
    total_videos = len(video_list)
    
    if total_videos == 0:
        print("CRITICAL: No videos found. Check your folder structures ('shaking/' and 'notshaking/').")
        return

    # 1. Define your hyperparameter search space
    search_space = {
        "target_height": [240], 
        "grid_rows": [6],
        "grid_cols": [8],
        "highpass_window": [10, 15, 20],
        "shake_threshold": [0.0008, 0.001, 0.0015, 0.002],
        "inlier_ratio_threshold": [0.65, 0.70, 0.75]
    }

    # Generate all possible parameter combinations
    keys, values = zip(*search_space.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Map each combo index to its running score across all videos
    # Initialize all combo scores to 0
    combo_scores = np.zeros(len(permutations), dtype=int)

    print("=" * 60)
    print(f"STARTING MEMORY-EFFICIENT SEQUENTIAL GRID SEARCH")
    print(f"Evaluating {len(permutations)} combinations across {total_videos} videos...")
    print("=" * 60)

    # 2. Outer Loop: Process one video at a time
    for v_idx, (video_path, expected, filename) in enumerate(video_list, 1):
        print(f"\n[{v_idx}/{total_videos}] Loading {filename} into RAM...")
        frames = load_single_video_to_ram(video_path)
        print("finished loading")
        if not frames:
            print(f"⚠️ Failed to read {filename}. Skipping.")
            continue
            
        total_frames = len(frames)
        
        # Inner Loop: Test every hyperparameter configuration on THIS specific video
        for c_idx, params in enumerate(permutations):
            detector = CameraShakeDetector(**params)
            shaking_frames = 0
            
            for frame in frames:
                result = detector.process_frame(frame)
                if result["is_shaking"]:
                    shaking_frames += 1
            
            # Evaluate prediction accuracy for this setup
            shake_ratio = shaking_frames / total_frames if total_frames > 0 else 0
            detected_shake = shake_ratio > 0.05
            
            if detected_shake == expected:
                combo_scores[c_idx] += 1
        
        # Garbage collect / drop frames list from memory implicitly before next loop iteration
        del frames 

        # Summary of the current leader after this video checkpoint
        current_max = np.max(combo_scores)
        print(f"-> Checked all configurations. Top running score so far: {current_max}/{v_idx}")

    # 3. Process Final Results
    best_combo_idx = np.argmax(combo_scores)
    best_score = combo_scores[best_combo_idx]
    best_params = permutations[best_combo_idx]

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    
    if best_score >= 22:
        print(f"🎉 SUCCESS! Found configurations matching target requirements.")
    else:
        print(f"⚠️ Search concluded. Target of 22 not quite met.")

    print(f"\nMax Score Achieved: {best_score} / {total_videos} Correct")
    print("Optimized Parameters:")
    print("-" * 60)
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print("-" * 60)

if __name__ == "__main__":
    run_optimization()