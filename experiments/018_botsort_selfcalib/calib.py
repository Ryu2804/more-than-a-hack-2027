"""018 stage A: can the missing tactical homography be recovered from the data itself?

Detect people (YOLO26, official Ultralytics COCO weights) on frames spread over the match, take the foot point (bottom centre of the box), and search for the
pitch->pixel homography that puts the ATD players (normalised pitch coordinates) on those foot points. No hand labels: random corner search scored by the number of
ATD players that land within a few pixels of a detection, then RANSAC/LM refinement. Reports inlier ratios and whether one homography fits all periods (static camera).
Outputs go to out/ (gitignored). Run: python calib.py [--n 40]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import common
from games import asset_paths

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
W_PX, H_PX = 1920, 1080


def detect(frames, model_name="yolo26m.pt"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    cap = cv2.VideoCapture(asset_paths()["video"])
    res = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, img = cap.read()
        if not ok: continue
        r = model.predict(img, imgsz=1920, conf=0.25, classes=[0], verbose=False, half=True)[0]
        b = r.boxes.xyxy.cpu().numpy()
        res[int(f)] = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]].astype(np.float32)   # foot points
    return res


def project(H, xy):
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def count_inliers(H, atd, dets, tol):
    from scipy.spatial import cKDTree
    tot = hit = 0
    for f, xy in atd.items():
        pr = project(H, xy)
        ok = np.isfinite(pr).all(1) & (pr[:, 0] > -50) & (pr[:, 0] < W_PX + 50) & (pr[:, 1] > -50) & (pr[:, 1] < H_PX + 50)
        d, _ = cKDTree(dets[f]).query(pr[ok]) if ok.any() and len(dets[f]) else (np.array([]), None)
        hit += int((d < tol).sum()); tot += len(xy)
    return hit / max(tot, 1)


def random_search(atd, dets, n_iter, rng, tol=15):
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])        # pitch corners: far-left, far-right, near-right, near-left (normalised)
    best = (0, None)
    for _ in range(n_iter):
        dst = np.float32([[rng.uniform(0.0, 0.45) * W_PX, rng.uniform(0.05, 0.5) * H_PX], [rng.uniform(0.55, 1.0) * W_PX, rng.uniform(0.05, 0.5) * H_PX],
                          [rng.uniform(0.8, 1.4) * W_PX, rng.uniform(0.6, 1.1) * H_PX], [rng.uniform(-0.4, 0.2) * W_PX, rng.uniform(0.6, 1.1) * H_PX]])
        H = cv2.getPerspectiveTransform(src, dst)
        s = count_inliers(H, atd, dets, tol)
        if s > best[0]: best = (s, H)
    return best


def refine(H, atd, dets, iters=30, tol=12):
    from scipy.spatial import cKDTree
    for t in range(iters):
        A, B = [], []
        for f, xy in atd.items():
            pr = project(H, xy); ok = np.isfinite(pr).all(1)
            if not ok.any() or not len(dets[f]): continue
            d, j = cKDTree(dets[f]).query(pr[ok]); keep = d < tol * (1 + 2 * (1 - t / iters))
            A.append(xy[ok][keep]); B.append(dets[f][j[keep]])
        A, B = np.concatenate(A), np.concatenate(B)
        if len(A) < 20: break
        H2, _ = cv2.findHomography(A.astype(np.float32), B.astype(np.float32), cv2.RANSAC, 6.0)
        if H2 is None: break
        H = H2
    return H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--iters", type=int, default=60000); a = ap.parse_args()
    P, B, T = common.load_atd_arrays()
    graded = np.r_[np.arange(11185, 90652), np.arange(110808, 190198)]
    rng = np.random.default_rng(0)
    frames = np.sort(rng.choice(graded, a.n, replace=False))
    dets = detect(frames)
    atd = {f: (P[f, :, :2][np.isfinite(P[f, :, 0])] * np.array([1.0, 1.0])).astype(np.float32) for f in dets}   # normalised 0-1 pitch coords
    first, second = [f for f in dets if f < 100000], [f for f in dets if f >= 100000]
    sub = lambda fs: ({f: atd[f] for f in fs}, {f: dets[f] for f in fs})
    out = {"frames": len(dets), "persons_per_frame": float(np.mean([len(v) for v in dets.values()])), "atd_players_per_frame": float(np.mean([len(v) for v in atd.values()]))}
    ta, td = sub(first[: len(first) // 2])                   # fit on half of the first half only, test on everything else
    s0, H0 = random_search(ta, td, a.iters, rng)
    out["search_score_fit"] = float(s0)
    H = refine(H0, ta, td) if H0 is not None else None
    if H is not None:
        out["fit_inlier"] = count_inliers(H, ta, td, 8)
        for name, fs in (("held_first_half", first[len(first) // 2:]), ("second_half", second)):
            xa, xd = sub(fs); out[f"{name}_inlier_8px"] = count_inliers(H, xa, xd, 8)
        np.save(OUT / "H_fit.npy", H)
    (OUT / "calib.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
