"""Notebook 00: re-derive the planning session's findings from cached exp_2 predictions.

IDSSE only. Uses ``exp2_aug``'s cached outer predictions on the ``target_like`` view decoded
with exp_2's own per-fold configs (selected on provider labels), so nothing is fitted or
re-selected here. Event-sync quality is reported from the open IDSSE event files.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import exp3_common  # noqa: F401,I001  (first: puts ../exp_2 and ../exp_1 on sys.path)
import exp1_decode as D
import exp1_protocol as P1
import exp2_views as V
import exp3_labels as L
import exp3_protocol as P
from exp1_common import PM, expand, load_clean, load_degraded, read_json, score
from exp3_common import EXP2, MATCH_IDS

SOURCE, VIEW = "exp2_aug", "target_like"
SHORT_S, MISS_SHORT_S, VISIBLE_SHARE = 2.0, 5.0, 0.8
FPS = 25


@lru_cache(maxsize=1)
def _predictions() -> dict:
    return P.cached_exp2(SOURCE)


def _selection(held: str) -> D.DecoderConfig:
    stored = read_json(EXP2 / "results" / f"variant_{SOURCE}__{VIEW}.json")["selections"][held]
    return D.DecoderConfig(**stored)


@lru_cache(maxsize=8)
def decoded(held: str) -> dict:
    """Outer prediction of one held-out match, decoded with exp_2's fold config (5 fps codes)."""
    probability, boundary = _predictions()[f"outer_{held}"][VIEW][held]
    item = P.make_item(held, (probability, boundary), VIEW, "provider")
    return {**item, "codes": D.decode(probability, boundary, item["half"], _selection(held))}


def _score(item: dict, codes: np.ndarray, truth: np.ndarray | None = None) -> dict:
    full = expand(codes.astype("int8"), item["indices"], item["full_half"])
    return score(item["truth"] if truth is None else truth, full)


def oracle_rows() -> list[dict]:
    """Per match: 5 fps ceiling, model, oracle dead/alive, oracle team (provider labels)."""
    rows = []
    for held in MATCH_IDS:
        item = decoded(held)
        state, codes, p = item["state"], item["codes"], item["probability"]
        oracle_dead = np.where(state == 2, 2, np.where(codes == 2, p[:, :2].argmax(1), codes))
        oracle_team = np.where((state < 2) & (codes < 2), state, codes)
        rows.append({"match": held,
                     "ceiling_5fps": _score(item, np.where(state < 3, state, 2))["spell_f1"],
                     "model": _score(item, codes)["spell_f1"],
                     "oracle_dead_alive": _score(item, oracle_dead)["spell_f1"],
                     "oracle_team_where_both_alive": _score(item, oracle_team)["spell_f1"]})
    return rows


def _matched(codes: np.ndarray, item: dict) -> tuple[list, np.ndarray]:
    truth = item["truth"]
    runs = PM.valid_runs(truth)
    full = PM.fill_abstentions(expand(codes.astype("int8"), item["indices"], item["full_half"]), runs)
    true_spells = PM.possession_spells(truth, runs)
    overlaps = PM.spell_overlaps(true_spells, PM.possession_spells(full, runs))
    return true_spells, PM.matched_gt_flags(overlaps, len(true_spells), 0.5)


def miss_rows() -> list[dict]:
    """Per true possession (provider): duration, matched at IoU 0.5 by the decoded / argmax
    output, ball-visible share on the target_like view, mean raw p(true team) over its steps."""
    rows = []
    for held in MATCH_IDS:
        item = decoded(held)
        visible = V.ball_visible(V.target_like(load_degraded(held, 0)))
        spells, decoded_flags = _matched(item["codes"], item)
        _, argmax_flags = _matched(item["probability"].argmax(1), item)
        for k, (start, stop, team) in enumerate(spells):
            steps = np.flatnonzero((item["indices"] >= start) & (item["indices"] < stop))
            rows.append({"match": held, "seconds": (stop - start) / FPS, "team": int(team),
                         "matched_decoded": bool(decoded_flags[k]), "matched_argmax": bool(argmax_flags[k]),
                         "ball_visible_share": float(visible[start:stop].mean()),
                         "mean_p_true_team": float(item["probability"][steps, team].mean()) if len(steps) else np.nan})
    return rows


def miss_summary(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    missed = frame[~frame.matched_decoded]
    short = frame[frame.seconds < SHORT_S]
    short_visible = short[short.ball_visible_share >= VISIBLE_SHARE]
    return {"true_possessions": int(len(frame)), "missed_iou05": int(len(missed)),
            "missed_share_under_5s": float((missed.seconds < MISS_SHORT_S).mean()),
            "share_of_possessions_under_2s": float((frame.seconds < SHORT_S).mean()),
            "recall_under_2s_decoded": float(short.matched_decoded.mean()),
            "recall_under_2s_argmax": float(short.matched_argmax.mean()),
            "missed_with_ball_visible_ge_80pct": float((missed.ball_visible_share >= VISIBLE_SHARE).mean()),
            "mean_p_true_team_under_2s_ball_visible": float(short_visible.mean_p_true_team.mean())}


def label_score_rows() -> list[dict]:
    """exp2_aug (exp_2 configs, no retraining, no re-selection) scored against provider and
    L_track(1, 2, 3 s). Not L_ctrl: under the fallback it equals track_2 (PREREG 0.4)."""
    rows = []
    for held in MATCH_IDS:
        item = decoded(held)
        for label_set in ("provider", "track_1", "track_2", "track_3"):
            rows.append({"match": held, "label_set": label_set,
                         **_score(item, item["codes"], P.frame_labels(held, label_set))})
    return rows


def sync_report() -> tuple[list[dict], dict]:
    """Event sync quality per match plus pooled shot residuals (kick-off only and snapped)."""
    rows, pooled = [], {"kickoff": [], "snapped": []}
    for name in MATCH_IDS:
        clean = load_clean(name)
        synced = L.sync_events(name, clean.frame, clean.half)
        events, report = P.control_events(name)
        pooled["kickoff"].append(L.shot_residuals(synced, clean.frame))
        pooled["snapped"].append(L.shot_residuals(events, clean.frame))
        changes = L.change_residuals(clean.labels, clean.half, events)
        finite = changes[np.isfinite(changes)]
        rows.append({"match": name, "events_synced": report["events_synced"],
                     "events_snapped": report["events_snapped"], "dropped_unlocated": report["dropped_unlocated"],
                     "shot_median_abs_kickoff": report["kickoff_sync"]["median_abs_frames"],
                     "shot_median_abs_snapped": report["snapped_sync"]["median_abs_frames"],
                     "shot_p90_abs_snapped": report["snapped_sync"]["p90_abs_frames"],
                     "change_median_abs_snapped": float(np.median(np.abs(finite))) if len(finite) else np.nan,
                     **{f"agree_{k}": v for k, v in L.possession_change_lag(clean.labels, events).items()}})
    return rows, {kind: L.sync_quality(np.concatenate(values)) for kind, values in pooled.items()}


def provider_counts() -> dict:
    out = {}
    for name in MATCH_IDS:
        truth = P1.proxy_truth(name)["truth"]
        out[name] = len(PM.possession_spells(truth, PM.valid_runs(truth)))
    return out
