# finish documenting function2.py
import cv2
import numpy as np
from second_iteration.functions2 import *
from collections import deque


def main(video_path=0, display=True):
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    detector = CameraShakeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        info = detector.process_frame(frame)

        if display:
            label = (
                f"Frame {info['frame_idx']:4d} | "
                f"tracks={info['n_tracks']:3d} | "
                f"dx={info['dx']:+.2f} dy={info['dy']:+.2f} "
                f"dθ={np.degrees(info['dtheta']):+.2f}° | "
                f"shake={info['shake_score']:.4f} "
                f"{'[SHAKING]' if info['is_shaking'] else ''}"
            )
            if(info['is_shaking']):
                print(label)

            # Optional visualization
            disp = frame.copy()
            color = (0, 0, 255) if info["is_shaking"] else (0, 255, 0)
            cv2.putText(
                disp,
                f"shake={info['shake_score']:.4f} {'SHAKING' if info['is_shaking'] else 'stable'}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.imshow("Shake Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else 0
    main(path)