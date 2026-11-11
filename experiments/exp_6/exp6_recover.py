"""BAR-ATH ball recovery end to end (PREREG 3.1-3.7).

Train the detector on train blocks, infer on hidden + val + test frames, link, choose tau on val,
measure quality once on test and coverage on hidden frames. Returns aggregates only; per-frame arrays
go to ``<work>/recovered_ball.npz`` (gitignored cache) and are never printed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Callable
import warnings

import numpy as np
import torch

import exp6_common as C  # noqa: F401  (environment first)
import exp6_data as DA
import exp6_detector as DT
import exp6_linker as LK
from exp1_common import JD

DISTANCE_BINS = (0.0, 1.0, 2.0, 5.0, np.inf)


def train_data(inputs: DA.TargetInputs) -> DT.TrainData:
    """Train-block frames with a kept auto-label; hidden, val and test frames never reach the sampler."""
    targets = (inputs.split == DA.TRAIN) & inputs.keep
    frames = np.flatnonzero(targets)
    u, v = DA.metres_to_pixels(inputs.matrices[frames], inputs.players[frames])
    return DT.TrainData(targets=targets, label_uv=inputs.label_uv, bbox=inputs.bbox,
                        players_uv=np.stack([u, v], axis=-1).astype("float32"),
                        image_size=(DA.IMAGE_W, DA.IMAGE_H))


def inference_mask(inputs: DA.TargetInputs) -> np.ndarray:
    return np.isin(inputs.split, (DA.HIDDEN, DA.VAL, DA.TEST))


def regions(bbox: np.ndarray, margin: int) -> np.ndarray:
    """Pitch bounding box plus ``margin`` pixels, clipped to the image."""
    out = bbox.astype("int64") + np.array([-margin, -margin, margin, margin])
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, DA.IMAGE_W)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, DA.IMAGE_H)
    return out


def runs_of(mask: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.array([stop - start for period in (1, 2) for start, stop in JD.runs(mask & (half == period))],
                    dtype=np.int64)


def run_summary(lengths: np.ndarray) -> dict:
    if not len(lengths):
        return {"runs": 0}
    return {"runs": int(len(lengths)), "frames": int(lengths.sum()), "mean": float(lengths.mean()),
            "median": float(np.median(lengths)), "p10": float(np.percentile(lengths, 10)),
            "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max()),
            "share_frames_in_runs_ge_25": float(lengths[lengths >= 25].sum() / lengths.sum())}


def _distance_summary(values: list[float]) -> dict:
    d = np.asarray(values, dtype=float)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)),
            "share_le_2m": float(np.mean(d <= 2.0)), "share_le_5m": float(np.mean(d <= 5.0))}


def bridge(inputs: DA.TargetInputs, position: np.ndarray, accepted: np.ndarray) -> dict:
    """Recovered position at a gap's first/last frame vs the ATD ball just outside the gap."""
    visible, half, starts, ends = inputs.visible, inputs.half, [], []
    for period in (1, 2):
        for start, stop in JD.runs((inputs.split == DA.HIDDEN) & (half == period)):
            if start > 0 and visible[start - 1] and half[start - 1] == period and accepted[start]:
                starts.append(float(np.linalg.norm(position[start] - inputs.ball[start - 1])))
            if stop < len(half) and visible[stop] and half[stop] == period and accepted[stop - 1]:
                ends.append(float(np.linalg.norm(position[stop - 1] - inputs.ball[stop])))
    return {"gap_start": _distance_summary(starts), "gap_end": _distance_summary(ends),
            "both": _distance_summary(starts + ends)}


def by_distance(inputs: DA.TargetInputs, frames: np.ndarray, accepted: np.ndarray, position: np.ndarray) -> list:
    """Test-block quality by distance from the ATD ball to the nearest ATD player."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gaps = np.linalg.norm(inputs.players[frames] - inputs.ball[frames][:, None], axis=-1)
        nearest = np.nanmin(gaps, axis=1)
    rows = []
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        keep = (nearest >= low) & (nearest < high)
        rows.append({"nearest_player_m": f"[{low:g}, {high:g})",
                     **LK.quality(accepted[keep], position[keep], inputs.ball[frames][keep])})
    return rows


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def recover(work: Path, log: Callable = print) -> dict:
    """Full pipeline into ``work``; resumable (detector checkpoints, inference chunks)."""
    cfg, link_cfg = DT.DetectorConfig(), LK.LinkerConfig()
    inputs = DA.load_inputs()
    data = train_data(inputs)
    reader = DA.VideoReader()
    start = time.time()
    model = DT.train(reader, data, cfg, work / "detector", log)
    trained = time.time()
    raw = DT.infer(model, reader, inference_mask(inputs), regions(inputs.bbox, cfg.infer_margin_px), cfg,
                   work / "infer", log)
    inferred = time.time()
    frames = raw["frame"]
    xy = DA.pixels_to_metres(np.linalg.inv(inputs.matrices[frames]), raw["uv"][..., 0], raw["uv"][..., 1])
    score = 1.0 / (1.0 + np.exp(-raw["logit"]))
    linked = LK.link(frames, xy, score, inputs.half, link_cfg)
    split, truth = inputs.split[frames], inputs.ball[frames]
    tau, tau_table = LK.select_tau(linked, truth, split == DA.VAL, link_cfg)
    accepted = LK.accept(linked, tau, link_cfg)
    test, val, hidden = split == DA.TEST, split == DA.VAL, split == DA.HIDDEN
    position_full = np.full((DA.N_FRAMES, 2), np.nan)
    position_full[frames] = linked["position"]
    accepted_full = np.zeros(DA.N_FRAMES, dtype=bool)
    accepted_full[frames] = accepted
    hidden_full = inputs.split == DA.HIDDEN
    lengths = np.sort(runs_of(accepted_full & hidden_full, inputs.half))
    arrays = {"frame": frames, "x": linked["position"][:, 0], "y": linked["position"][:, 1],
              "score": linked["score"], "accepted": accepted, "split": split}
    np.savez(work / "recovered_ball.npz", **arrays, tau=np.array(tau))
    np.save(work / "run_lengths.npy", lengths)
    graded = inputs.half > 0
    return {
        "detector": {"config": cfg.as_dict(), "train_frames": int(data.targets.sum()),
                     "samples_per_pass": DT.samples_per_pass(data, cfg), "state_sha256": state_digest(model),
                     "train_seconds": round(trained - start, 1), "inference_seconds": round(inferred - trained, 1),
                     "inferred_frames": int(len(frames))},
        "linker": link_cfg.as_dict(), "tau": tau, "tau_table_val": tau_table,
        "val": LK.quality(accepted[val], linked["position"][val], truth[val]),
        "test": LK.quality(accepted[test], linked["position"][test], truth[test]),
        "coverage": float(accepted[hidden].mean()), "hidden_frames": int(hidden.sum()),
        "accepted_hidden_frames": int(accepted[hidden].sum()),
        "coverage_by_tau": {str(t): float(LK.accept(linked, t, link_cfg)[hidden].mean()) for t in LK.TAU_GRID},
        "on_path_share_hidden": float((linked["state"] < linked["k"])[hidden].mean()),
        "run_lengths": run_summary(lengths), "bridge": bridge(inputs, position_full, accepted_full),
        "test_by_nearest_player": by_distance(inputs, frames[test], accepted[test], linked["position"][test]),
        "ball_visibility_graded": {"atd": float(inputs.visible[graded].mean()),
                                   "atd_plus_recovered": float((inputs.visible | (accepted_full & hidden_full))[graded].mean())},
        "recovered_sha256": array_digest(arrays),
    }
