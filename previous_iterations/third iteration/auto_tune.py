import os, glob, csv, itertools, time
from shake_detector import detect_shake_in_video

SHAKING_DIR    = "shaking"
NOTSHAKING_DIR = "notshaking"
VIDEO_EXTS     = ("*.mp4", "*.mov", "*.avi", "*.mkv")

# ---- DEFINE YOUR SWEEP ----
SWEEP = {
    "sensitivity":       [0.05, 0.1, 0.15, 0.2],
    "qopts":             [0.1],
    "sensitivity2":      [0.7, 0.8, 0.9],
    "spread":            [10],                       # fixed — grid caps this anyway
    "minpts":            [40, 60, 80],
    "shake_frame_ratio": [0.05, 0.10],
    "frame_skip":        [2],                        # fixed — speed only
    "pts_per_cell":      [2],                        # fixed for now
}
# Optimization target: "accuracy", "f1", or "balanced"
SCORE = "balanced"
# ----------------------------

def collect(folder, label):
    out = []
    for ext in VIDEO_EXTS:
        for p in glob.glob(os.path.join(folder, ext)):
            out.append((p, label))
    return sorted(out)

def score_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    balanced = (recall + specificity) / 2
    return dict(accuracy=acc, precision=precision, recall=recall,
                specificity=specificity, f1=f1, balanced=balanced,
                tp=tp, fp=fp, tn=tn, fn=fn)

def evaluate(params, videos):
    tp = fp = tn = fn = 0
    for path, truth in videos:
        pred, _, _ = detect_shake_in_video(path, **params)
        if pred is None: continue
        if   truth and pred:     tp += 1
        elif truth and not pred: fn += 1
        elif not truth and pred: fp += 1
        else:                    tn += 1
    return score_metrics(tp, fp, tn, fn)

def main():
    videos = collect(SHAKING_DIR, True) + collect(NOTSHAKING_DIR, False)
    keys = list(SWEEP.keys())
    combos = list(itertools.product(*SWEEP.values()))
    print(f"{len(videos)} videos x {len(combos)} param combos "
          f"= {len(videos)*len(combos)} runs. Scoring on '{SCORE}'.\n")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        m = evaluate(params, videos)
        dt = time.time() - t0
        row = {**params, **m, "time_s": round(dt, 1)}
        results.append(row)
        print(f"[{i}/{len(combos)}] {params} -> "
              f"{SCORE}={m[SCORE]:.2%} acc={m['accuracy']:.2%} "
              f"TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']} "
              f"({dt:.0f}s)")

    with open("tuning_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    results.sort(key=lambda r: r[SCORE], reverse=True)
    print(f"\n--- Top 5 by {SCORE} ---")
    for r in results[:5]:
        print(r)
    print("\nFull results saved to tuning_results.csv")

if __name__ == "__main__":
    main()