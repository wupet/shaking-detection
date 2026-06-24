import os
import cv2
from previous_iterations.second_iteration.functions2 import CameraShakeDetector

def evaluate_video(video_path, detector_params):
    """
    Processes a single video file frame-by-frame.
    Returns True if shaking was detected consistently/frequently enough, else False.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Initialize a clean detector instance for each video
    detector = CameraShakeDetector(**detector_params)
    
    shaking_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        result = detector.process_frame(frame)
        
        if result["is_shaking"]:
            shaking_frames += 1

    cap.release()

    # --- ADJUSTABLE EVALUATION CRITERIA ---
    # If more than 5% of the total frames registered shaking, flag the video.
    # Alternatively, you can use a raw frame count like: if shaking_frames > 15
    shake_ratio = shaking_frames / total_frames if total_frames > 0 else 0
    detected_shake = shake_ratio > 0.01 
    
    return detected_shake, shaking_frames, total_frames


def run_batch_test():
    # 1. Define your detector hyperparameters here to tune them easily
    detector_params = {
        "target_height": 240,
        "highpass_window": 16,
        "shake_threshold": 0.001,      # Lower this if small shakes are missed
        "inlier_ratio_threshold": .95   # increase this if it is detecting other movement as shakes
    }

    # 2. Configure path structures
    # Assuming the folders "shaking" and "not shaking" sit next to this script
    categories = {
        "shaking": {"folder": "shaking", "prefix": "shaking", "expected": True},
        "not shaking": {"folder": "notshaking", "prefix": "notshaking", "expected": False}
    }

    results = []
    print("=" * 60)
    print("STARTING CAMERA SHAKE DETECTOR AUTOMATED BATCH TEST")
    print("=" * 60)

    for category_name, config in categories.items():
        print(f"\nEvaluating Category: [{category_name.upper()}] (Expecting 'is_shaking' == {config['expected']})")
        print("-" * 60)
        
        for i in range(1, 13):
            filename = f"{config['prefix']}{i}.mp4"
            video_path = os.path.join(config["folder"], filename)
            
            if not os.path.exists(video_path):
                print(f"File missing: {video_path} ... Skipping.")
                continue

            # Run evaluation
            detected, shake_count, total = evaluate_video(video_path, detector_params)
            
            if detected is None:
                continue

            # Calculate match accuracy
            success = (detected == config["expected"])
            status_str = "PASS" if success else "FAIL"
            
            results.append({
                "video": filename,
                "expected": config["expected"],
                "detected": detected,
                "success": success
            })
            
            print(f"-> {filename:<18} | Shaking Frames: {shake_count:>3}/{total:<3} | Detected: {str(detected):<5} | [{status_str}]")

    # 3. Print Final Analytics Summary Report
    print("\n" + "=" * 60)
    print("FINAL SUMMARY STATISTICS")
    print("=" * 60)
    
    total_tested = len(results)
    total_passed = sum(1 for r in results if r["success"])
    
    # Calculate performance matrices
    true_positives = sum(1 for r in results if r["expected"] and r["detected"])
    false_negatives = sum(1 for r in results if r["expected"] and not r["detected"])
    true_negatives = sum(1 for r in results if not r["expected"] and not r["detected"])
    false_positives = sum(1 for r in results if not r["expected"] and r["detected"])

    accuracy = (total_passed / total_tested) * 100 if total_tested > 0 else 0
    
    print(f"Total Videos Tested: {total_tested}")
    print(f"Total Videos Passed: {total_passed} / {total_tested}")
    print(f"Overall Accuracy:    {accuracy:.2f}%")
    print("-" * 60)
    print(f"True Positives  (Correctly caught shaking): {true_positives}")
    print(f"False Negatives (Missed actual shaking):    {false_negatives}")
    print(f"True Negatives  (Correctly ignored still):  {true_negatives}")
    print(f"False Positives (Falsely flagged still):   {false_positives}")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_test()