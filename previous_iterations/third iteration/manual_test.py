import os, glob, time
from shake_detector import detect_shake_in_video

# ---- TUNE THESE BY HAND ----
PARAMS = dict(
    sensitivity        = 0.2, 
    qopts              = 0.01,
    sensitivity2       = 0.7,
    spread             = 5,
    minpts             = 100,
    fb_error_threshold = 0.5,
    shake_frame_ratio  = 0.01,
    frame_skip         = 1,
    grid_cols          = 16,
    grid_rows          = 12,
    pts_per_cell       = 1,
)
# ----------------------------

SHAKING_DIR    = "shaking"
NOTSHAKING_DIR = "notshaking"
VIDEO_EXTS     = ("*.mp4", "*.mov", "*.avi", "*.mkv")

def collect(folder, label):
    out = []
    for ext in VIDEO_EXTS:
        for p in glob.glob(os.path.join(folder, ext)):
            out.append((p, label))
    return sorted(out)

def main():
    videos = collect(SHAKING_DIR, True) + collect(NOTSHAKING_DIR, False)
    print(f"Found {len(videos)} videos.\nParams: {PARAMS}\n")

    tp = fp = tn = fn = 0
    misses = []

    for path, truth in videos:
        t0 = time.time()
        pred, ratio, frames = detect_shake_in_video(path, **PARAMS)
        dt = time.time() - t0

        if pred is None:
            print(f"  [SKIP] {path} (could not open)")
            continue

        mark = "OK  " if pred == truth else "MISS"
        t_str = "shake" if truth else "still"
        p_str = "shake" if pred  else "still"
        print(
              f"  [{mark}] {os.path.basename(path):<25} "
            #   f"truth={t_str} pred={p_str} "
              f"ratio={ratio:.2%} ({frames}f, {dt:.1f}s)")

        if   truth and pred:       tp += 1
        elif truth and not pred:   fn += 1; misses.append(path)
        elif not truth and pred:   fp += 1; misses.append(path)
        else:                      tn += 1
        if path == "shaking\\shaking9.mp4":
            print()
            
    total = tp + fp + tn + fn
    if total == 0:
        print("\nNo videos scored."); return

    print(f"\n--- Results ---")
    print(f"Accuracy: {(tp+tn)/total:.2%}  ({tp+tn}/{total})")
    print(f"Shaking  caught: {tp}/{tp+fn}   (FN={fn})")
    print(f"Still    passed: {tn}/{tn+fp}   (FP={fp})")
    if misses:
        print("\nMisclassified:")
        for m in misses: print(f"  {m}")

if __name__ == "__main__":
    main()