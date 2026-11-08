"""YOLO ball-detection feasibility on match_only.mp4 (see NOTES.md). Usage: python feasibility.py <outdir> <weights.pt> <imgsz>"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path[:0] = [ROOT + "/experiments", ROOT, ROOT + "/starter"]
import numpy as np, cv2, common
from games import half_video_blocks
from ultralytics import YOLO

CLIP = ROOT + "/data/match_only.mp4"
OUT = sys.argv[1]; WEIGHTS = sys.argv[2]; IMGSZ = int(sys.argv[3])
FPS = 25
P, B, T = common.load_atd_arrays(); vis = np.isfinite(B[:, 0]); N = len(B)
blocks = half_video_blocks("BAR-ATH")
def to_clip(f):
    (_, a1, b1), (_, a2, b2) = blocks
    return f - a1 if f <= b1 else (b1 - a1 + 1) + (f - a2)

rng = np.random.default_rng(0)
# ball-missing snippets: mid-point of gaps >= 10 s;  ball-visible snippets: >= 4 s fully visible
gap, live = [], []
for h, a, b in blocks:
    s, e = common.runs(~vis[a:b + 1])
    for si, ei in zip(s, e):
        if (ei - si) >= 10 * FPS: gap.append(a + si + (ei - si) // 2 - 50)
    s, e = common.runs(vis[a:b + 1])
    for si, ei in zip(s, e):
        if (ei - si) >= 6 * FPS: live.append(a + si + 25)
gap = list(rng.choice(gap, 8, replace=False)); live = list(rng.choice(live, 8, replace=False))
plan = [("gap", f) for f in gap] + [("vis", f) for f in live]

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CLIP)
rows = []; saved = {"gap": 0, "vis": 0}
for kind, f0 in plan:
    cap.set(cv2.CAP_PROP_POS_FRAMES, to_clip(f0))
    frames = []
    for k in range(100):
        ok, fr = cap.read()
        if not ok: break
        if k % 4 == 0: frames.append((f0 + k, fr))
    res = [model.predict(fr, imgsz=IMGSZ, conf=0.03, classes=[0, 32], verbose=False, half=True, device=0)[0] for _, fr in frames]   # one frame at a time (a list would be one batch)
    for (fi, fr), r in zip(frames, res):
        cls = r.boxes.cls.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); xy = r.boxes.xywh.cpu().numpy()
        balls = [(c, x) for c, k_, x in zip(cf, cls, xy) if int(k_) == 32]
        best = max(balls, key=lambda t: t[0]) if balls else None
        rows.append(dict(kind=kind, f=int(fi), n_players=int((cls == 0).sum()), n_ball=len(balls),
                         conf=float(best[0]) if best else 0.0, w=float(best[1][2]) if best else 0.0, h=float(best[1][3]) if best else 0.0,
                         cx=float(best[1][0]) if best else -1, cy=float(best[1][1]) if best else -1))
        # crude on-pitch test: fraction of green pixels in a 61x61 window around the detection centre
        onpitch = False
        if best:
            cx, cy = int(best[1][0]), int(best[1][1]); x0, y0 = max(cx - 30, 0), max(cy - 30, 0)
            hsv = cv2.cvtColor(fr[y0:cy + 31, x0:cx + 31], cv2.COLOR_BGR2HSV)
            green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 60)).mean() if hsv.size else 0
            onpitch = bool(green > 0.5)
        rows[-1]["onpitch"] = onpitch
        if best and best[0] > 0.2 and saved[kind] < 12 and fi % 8 == 0:
            cx, cy = int(best[1][0]), int(best[1][1]); x0, y0 = max(cx - 60, 0), max(cy - 60, 0)
            crop = cv2.resize(fr[y0:cy + 61, x0:cx + 61], (360, 360), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f"{kind} f{fi} c{best[0]:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imwrite(f"{OUT}/crop_{kind}_{fi}.jpg", crop)
        if best and saved[kind] < 4 and best[0] > 0.2:
            im = fr.copy(); x, y, w, h = best[1]
            cv2.rectangle(im, (int(x - w), int(y - h)), (int(x + w), int(y + h)), (0, 0, 255), 3)
            cv2.imwrite(f"{OUT}/{kind}_{fi}.jpg", im); saved[kind] += 1
json.dump(rows, open(f"{OUT}/rows.json", "w"))
import pandas as pd
d = pd.DataFrame(rows)
print(f"weights {WEIGHTS} imgsz {IMGSZ} | frames: {d.groupby('kind').size().to_dict()} | players/frame: {d.n_players.mean():.1f}")
for kind, g in d.groupby("kind"):
    for th in (0.05, 0.2, 0.4):
        print(f"{kind}: frames with ball detection conf>={th}: {(g.conf >= th).mean():.1%}")
    op = g[(g.conf >= 0.2) & (g.onpitch)]
    print(f"{kind}: ball detections conf>=0.2 whose surroundings are mostly pitch-green: {len(op) / len(g):.1%} of frames")
    gg = g[g.conf >= 0.2]
    print(f"{kind}: median bbox w,h (px) of detections conf>=0.2: {gg.w.median():.1f}, {gg.h.median():.1f} | n={len(gg)}")
