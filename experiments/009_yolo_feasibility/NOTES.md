# 009 yolo feasibility (negative result)

**Hypothesis:** a pretrained YOLO ball detector run on ball-missing frames of `data/match_only.mp4` can recover the ball and feed the Viterbi decoder.

**Setup:** `feasibility.py`: 8 ball-missing snippets (middle of ATD gaps >= 10 s) + 8 ball-visible snippets, 25 frames each (stride 4), YOLOv8m COCO weights (classes person, sports ball), imgsz 1920, conf >= 0.03, fp16 on the RTX 4060. Annotated frames written outside the repo (they are derived match images, not committed).

**Result:** detection "rates" look high and are meaningless: ball box with conf >= 0.2 on 82% of gap frames and 94% of visible frames; 61 "players" per frame (crowd, staff). Inspection of annotated frames: on this fixed wide camera the real ball is ~2-4 px; in both inspected frames the real ball (centre spot at a kick-off; a dot near a player) was missed and the detections were false positives on advertising boards / stands. Median detection box 8-12 px = not the ball.

**Verdict:** generic COCO YOLO is unusable here (ball far below its object-size range). A usable detector would need (a) a ball-specific tiny-object model trained on tactical/wide-camera football footage from other matches (labels not available here), (b) a pitch mask (needs the tactical homography, still missing) to remove stands/boards, and (c) temporal filtering. Not pursued. Side note: the video has a burned-in match clock and score (top-left); the clock runs through stoppages, so it does not indicate dead ball.

**Environment change:** `pip install ultralytics` (ultralytics 8.4.202; torch unchanged 2.12.1+cu130; setuptools 84 -> 81; added opencv-python 5.0.0.93).

---

## Update: YOLO26 (docs.ultralytics.com/models/yolo26) changes the verdict

Same 400 frames (200 ATD-ball-missing, 200 ATD-ball-visible), imgsz 1920, frames fed one at a time (a list = one batch; `yolo26x` OOMs at batch 25 on the 8 GB RTX 4060).

| model | ball conf>=0.2 on ATD-missing frames | on ATD-visible frames |
|---|---|---|
| yolov8m (COCO) | 82.0% | 93.5% (mostly false positives: boards, stands, pitch marks) |
| yolo26x (COCO) | 23.5% | 66.0% |

The contrast (23.5% vs 66%) is what a real ball detector would show. Zoomed crops (not committed, derived match images) confirm: in ATD-visible frames the boxes sit on the actual ball at a player's feet; in ATD-missing frames YOLO26x finds a ball lying still at the touchline beside staff (a stoppage the tracker lost). Detection boxes are ~7-8 px. Residual false positives exist (e.g. near the goal net), so temporal and pitch-geometry filtering would still be needed.

Speed at 1920 px, one frame at a time: yolo26x 72 ms, yolo26l 39 ms, yolo26m 32 ms per frame. On a 40-frame visible snippet the hit counts were similar (x 33, l 36, m 37 of 40), so yolo26m/l is the efficient choice. All 68k ATD-missing graded frames: x 1.4 h, m 0.6 h; at stride 5: x 17 min, m 7 min.

**Blocker for using it:** detections are in pixels and the camera pans/zooms, so they need the tactical homography (`data/tactical/BAR-ATH_tactical_homography.csv`, still not downloaded) to become pitch positions for the nearest-player / control-radius evidence and for dead-ball cues (ball stationary at the line). Alternative: fit a per-frame homography from ATD players to detected people; heavier and less reliable.
