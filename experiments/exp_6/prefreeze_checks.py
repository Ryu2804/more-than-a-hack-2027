"""Checks allowed before freezing (PROMPT "Before freezing"); aggregates only.

* feasibility by import: reused files exist, exp_2 fingerprints, stored selections, units;
* G1 and split counts from ATD + homography only (no frame decoded);
* a decode-speed test that reads 300 frames (timing, frame count, determinism match only);
* U-Net memory / throughput on random tensors.

Writes ``results/prefreeze_checks.json``. No detector training on BAR-ATH, no simulated-view scoring.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import exp6_common as C
import exp1_common as C1
import exp2_protocol as P2
import exp3_coverage as V3  # noqa: F401  (import check: S2 recheck reuses coverage_view)
import exp6_data as DA
import exp6_detector as DT


def feasibility() -> dict:
    files = C.reused_files()
    missing = sorted(key for key, path in files.items() if not path.exists())
    selections = {label: sorted(C1.read_json(C.exp3_result(C.REFERENCE, label))["selections"])
                  for label in C.LABEL_SETS}
    clean = C1.load_clean(C.MATCH_IDS[0])
    finite = np.isfinite(clean.ball).all(axis=1)
    return {"reused_files": len(files), "missing": missing,
            "exp2_fingerprints": {arm: P2.fingerprint(arm) for arm in C.MEMBERS},
            "exp2_folders": {arm: P2.prediction_folder(arm).name for arm in C.MEMBERS},
            "exp3_selection_matches": {k: v == sorted(C.MATCH_IDS) for k, v in selections.items()},
            "idsse_units_check": {"match": clean.name, "pitch": list(clean.pitch),
                                  "ball_x_p99_over_length": float(np.percentile(clean.ball[finite, 0], 99) / clean.pitch[0]),
                                  "ball_y_p99_over_width": float(np.percentile(clean.ball[finite, 1], 99) / clean.pitch[1])}}


def gpu_test(cfg: DT.DetectorConfig, region_hw: tuple[int, int]) -> dict:
    model = DT.new_model(cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x = torch.randint(0, 256, (cfg.batch, 9, cfg.crop, cfg.crop), dtype=torch.uint8, device=DT.DEVICE)
    offsets = torch.rand((cfg.batch, 2), device=DT.DEVICE) * cfg.crop
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(12):
        torch.cuda.synchronize()
        start = time.time()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x.float() / 255), DT.heatmap(offsets, cfg.crop, cfg.sigma_px),
            pos_weight=torch.tensor(cfg.pos_weight, device=DT.DEVICE))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    train_peak = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    h, w = (int(np.ceil(s / cfg.stride) * cfg.stride) for s in region_hw)
    frame = torch.randint(0, 256, (1, 9, h, w), dtype=torch.uint8, device=DT.DEVICE)
    torch.cuda.reset_peak_memory_stats()
    infer = []
    with torch.no_grad():
        for _ in range(8):
            torch.cuda.synchronize()
            start = time.time()
            logits = model(frame.float() / 255)[0, 0]
            DT.peaks(logits, cfg.top_k, cfg.nms)
            torch.cuda.synchronize()
            infer.append(time.time() - start)
    return {"parameters": int(sum(p.numel() for p in model.parameters())),
            "train_step_ms_batch64": round(1000 * float(np.median(times[2:])), 1),
            "train_peak_mib": round(train_peak, 1), "inference_input_hw": [h, w],
            "inference_ms_per_frame": round(1000 * float(np.median(infer[2:])), 1),
            "inference_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def main() -> None:
    out = {"feasibility": feasibility()}
    print("feasibility:", out["feasibility"], flush=True)
    inputs = DA.load_inputs()
    out["inputs"] = DA.input_summary(inputs)
    margin = DT.DetectorConfig().infer_margin_px
    box = inputs.bbox[inputs.half > 0]
    out["region_median_hw"] = [float(np.median(np.minimum(box[:, 3] + margin, DA.IMAGE_H) - np.maximum(box[:, 1] - margin, 0))),
                               float(np.median(np.minimum(box[:, 2] + margin, DA.IMAGE_W) - np.maximum(box[:, 0] - margin, 0)))]
    ball_ok = inputs.visible & (inputs.half > 0)
    inside = ((inputs.ball[ball_ok] >= 0) & (inputs.ball[ball_ok] <= np.array(DA.PITCH))).all(axis=1)
    out["barath_units_check"] = {"visible_graded_inside_pitch_share": float(inside.mean())}
    print("inputs:", out["inputs"], out["region_median_hw"], out["barath_units_check"], flush=True)
    del inputs
    reader = DA.VideoReader()
    out["video_metadata"] = reader.metadata()
    out["decode"] = DA.decode_check(reader)
    print("video:", out["video_metadata"], out["decode"], flush=True)
    out["gpu"] = gpu_test(DT.DetectorConfig(), tuple(int(v) for v in out["region_median_hw"]))
    print("gpu:", out["gpu"], flush=True)
    C1.write_json(C.RESULTS / "prefreeze_checks.json", out)


if __name__ == "__main__":
    main()
