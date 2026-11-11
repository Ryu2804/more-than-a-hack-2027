"""Proxy simulator S(c, sigma, f) (PREREG 3.8-3.9) and scoring of ``exp3_ref_ens3``.

Scoring uses exp_2's saved outer-fold weights and exp_3's stored per-fold ``target_like`` decoder
selections (no fit, no re-selection). Also: the two simulator rechecks and the ``exp5_touch_ens3``
context (report only). IDSSE data only; BAR-ATH enters only through the run-length distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Callable

import numpy as np
import torch

import exp6_common as C
import exp1_common as C1
import exp1_decode as D
import exp1_net as N
import exp1_protocol as P1
import exp1_tokens as T
import exp2_protocol as P2
import exp2_views as V
import exp3_coverage as V3
import exp3_protocol as P3
from exp1_common import JD, MATCH_IDS

SEED_OFFSET = 7000
FP_NOISE_M = 1.0  # false-positive run: player position + N(0, 1 m^2) per axis
STRESS_RULE = {"coverage": "c_hat / 2", "noise": "2 * sigma_hat", "fp": "min(1, 2 * f_hat + 0.05)"}
VOI_COVERAGE = (0.12, 0.25, 0.5, 1.0)
ROW_KEYS = ("spell_f1", "macro_f1", "n_true", "n_pred", "count_error_ratio")
S2_RECHECK = "exp3_s2_coverage"


@dataclass(frozen=True)
class Setting:
    name: str
    coverage: float
    noise: float
    fp: float

    def as_dict(self) -> dict:
        return asdict(self)


BASE = Setting("target_like", 0.0, 0.0, 0.0)


def _finite(value: float | None, default: float) -> float:
    return float(value) if value is not None and np.isfinite(value) else default


def _measured_values(q: dict) -> tuple[float, float, float]:
    """(c, sigma, f); an undefined sigma counts as 0 and an undefined f as 1 (no accepted test frame)."""
    return _finite(q["coverage"], 0.0), _finite(q["sigma_m"], 0.0), _finite(q["fp_rate"], 1.0)


def measured(q: dict) -> Setting:
    return Setting("measured", *_measured_values(q))


def stress(q: dict) -> Setting:
    c, sigma, fp = _measured_values(q)
    return Setting("stress", c / 2, 2 * sigma, min(1.0, 2 * fp + 0.05))


def voi_grid(q: dict) -> list[Setting]:
    """{coverage} x {0, sigma_hat} x {0, f_hat}; cells that coincide (sigma_hat or f_hat = 0) appear once."""
    _, sigma, fp = _measured_values(q)
    cells = dict.fromkeys((c, s, f) for c in VOI_COVERAGE for s in (0.0, sigma) for f in (0.0, fp))
    return [Setting(f"voi_c{c:g}_s{'hat' if s else 0}_f{'hat' if f else 0}", c, s, f) for c, s, f in cells]


def sim_seed(name: str) -> int:
    return C.BASE_SEED + SEED_OFFSET + MATCH_IDS.index(name)


# ---------------------------------------------------------------- simulator
def hidden_frames(base: JD.Match, clean_ball: np.ndarray) -> np.ndarray:
    """Graded frames with no ball in the base view but a ball in the clean match."""
    return (base.half > 0) & ~V.ball_visible(base) & np.isfinite(clean_ball).all(axis=1)


def _starts(free: np.ndarray, length: int) -> np.ndarray:
    cumulative = np.r_[0, np.cumsum(free)]
    return np.flatnonzero(cumulative[length:] - cumulative[:-length] == length)


def place_subruns(free: np.ndarray, target: int, lengths: np.ndarray,
                  rng: np.random.Generator) -> list[tuple[int, int]]:
    """Non-overlapping sub-runs inside ``free`` until ``target`` frames; lengths drawn from ``lengths``,
    the last truncated; a draw longer than every free stretch is truncated to the longest one."""
    free = free.copy()
    placed, revealed = [], 0
    while revealed < target:
        length = min(int(rng.choice(lengths)), target - revealed)
        starts = _starts(free, length)
        if not len(starts):
            length = max(stop - start for start, stop in JD.runs(free))
            starts = _starts(free, length)
        start = int(starts[rng.integers(len(starts))])
        free[start:start + length] = False
        placed.append((start, length))
        revealed += length
    return placed


def _player_track(players: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random player present on every frame of the sub-run (else the most present one)."""
    finite = np.isfinite(players).all(axis=2)
    complete = np.flatnonzero(finite.all(axis=0))
    slot = int(complete[rng.integers(len(complete))]) if len(complete) else int(finite.sum(axis=0).argmax())
    return players[:, slot]


def simulate(base: JD.Match, clean: JD.Match, setting: Setting, lengths: np.ndarray,
             rng: np.random.Generator) -> tuple[JD.Match, dict]:
    """``base`` with the clean ball re-revealed on sub-runs of hidden frames (noise, FP runs)."""
    hidden = hidden_frames(base, clean.ball)
    target = int(round(setting.coverage * hidden.sum()))
    ball = base.ball.astype("float64")
    subruns = place_subruns(hidden, target, lengths, rng) if target else []
    fp_frames = 0
    for start, length in subruns:
        frames = slice(start, start + length)
        if rng.random() < setting.fp:
            track = _player_track(clean.players[frames, :, :2].astype("float64"), rng)
            ball[frames] = track + rng.normal(0.0, FP_NOISE_M, (length, 2))
            fp_frames += length
        else:
            ball[frames] = clean.ball[frames] + rng.normal(0.0, setting.noise, (length, 2))
    view = replace(base, ball=ball.astype(base.ball.dtype))
    graded = base.half > 0
    stats = {"hidden_frames": int(hidden.sum()), "target_frames": target, "subruns": len(subruns),
             "revealed_frames": int(sum(length for _, length in subruns)), "fp_frames": fp_frames,
             "revealed_share_of_hidden": float(target / max(1, hidden.sum())),
             "ball_visible_before": float(V.ball_visible(base)[graded].mean()),
             "ball_visible_after": float(V.ball_visible(view)[graded].mean())}
    return view, stats


def make_view(setting: Setting, base: JD.Match, clean: JD.Match, lengths: np.ndarray,
              name: str) -> tuple[JD.Match, dict]:
    if setting.name == S2_RECHECK:  # exp_3's own S2 coverage view, regenerated in memory
        return V3.coverage_view(C1.load_degraded(name, 0), clean.ball,
                                np.random.default_rng(V3.reveal_seed(name)))
    return simulate(base, clean, setting, lengths, np.random.default_rng(sim_seed(name)))


# ---------------------------------------------------------------- saved-weight scoring
def token_item(view: JD.Match) -> dict:
    token_set = T.build_tokens(view)
    return {"tokens": token_set.tokens, "mask": token_set.mask, "glob": token_set.glob,
            "half": token_set.half, "indices": token_set.indices}


def load_member(arm: str, held: str) -> N.SetSeq:
    """exp_2 outer-fold member (the pattern of exp3_protocol._member_model)."""
    config = P2.MODEL_ARMS[arm]["config"]
    model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
    model.load_state_dict(torch.load(C.exp2_outer_weight(arm, held), map_location=N.DEVICE))
    return model.eval()


def load_touch_member(arm: str, held: str) -> N.SetSeq:
    """exp_5 touch member: SetSeq with exp_3's 4 extra globals; the unused aux head is dropped."""
    import exp3_touch as S3T

    state = torch.load(C.exp5_touch_folder(arm) / f"outer_{held}.pt", map_location=N.DEVICE)
    state = {key: value for key, value in state.items() if not key.startswith("aux_head.")}
    config = N.TrainConfig()
    with S3T.extended_globals():
        model = N.SetSeq(config.width, config.hidden).to(N.DEVICE)
    model.load_state_dict(state)
    return model.eval()


def touch_item(view: JD.Match, item: dict) -> dict:
    import exp3_touch as S3T

    return {**item, "glob": np.concatenate([item["glob"], S3T.touch_features(view).astype("float16")], axis=1)}


FAMILIES = {
    "reference": {"members": C.MEMBERS, "loader": load_member, "result": lambda label: C.exp3_result(C.REFERENCE, label)},
    "touch": {"members": C.TOUCH_MEMBERS, "loader": load_touch_member, "result": C.exp5_result},
}


def ensemble(models: list[N.SetSeq], item: dict, window: int) -> tuple[np.ndarray, np.ndarray]:
    parts = [N.predict(model, item, window) for model in models]
    return tuple(np.mean([p[i] for p in parts], axis=0).astype("float32") for i in (0, 1))


def score_prediction(name: str, prediction: tuple, item: dict, label_set: str, config: dict) -> dict:
    """Decode with one fold's stored selection and score once with the official metric."""
    if not np.array_equal(item["indices"], P2.load_tokens(name)["indices"]):
        raise RuntimeError("View tokens are not aligned with the original grid")
    state, _ = P3.step_labels(name, label_set, item)
    full = {"probability": prediction[0], "boundary": prediction[1], "half": item["half"],
            "indices": item["indices"], "state": state, "truth": P3.frame_labels(name, label_set),
            "full_half": P1.proxy_truth(name)["full_half"]}
    return D.score_item(full, D.DecoderConfig(**config))


def cached_base_diff(name: str, prediction: tuple) -> float:
    """Max |p| difference vs exp_2's cached target_like ensemble (diagnostic)."""
    parts = []
    for arm in C.MEMBERS:
        with np.load(P2.prediction_folder(arm) / f"outer_{name}.npz") as z:
            parts.append((z[f"target_like__{name}_probability"], z[f"target_like__{name}_boundary"]))
    cached = tuple(np.mean([p[i] for p in parts], axis=0).astype("float32") for i in (0, 1))
    return float(max(np.abs(cached[i] - prediction[i]).max() for i in (0, 1)))


def evaluate(settings: list[Setting], lengths: np.ndarray, family: str = "reference",
             log: Callable = print) -> list[dict]:
    """Rows (family, setting, match, label set, view stats, official scores) for every setting."""
    spec = FAMILIES[family]
    selections = {label: C1.read_json(spec["result"](label))["selections"] for label in C.LABEL_SETS}
    window = N.TrainConfig().window
    rows = []
    for name in MATCH_IDS:
        base, clean = V.target_like(C1.load_degraded(name, 0)), C1.load_clean(name)
        models = [spec["loader"](arm, name) for arm in spec["members"]]
        for setting in settings:
            start = time.time()
            view, stats = make_view(setting, base, clean, lengths, name)
            item = token_item(view)
            if family == "touch":
                item = touch_item(view, item)
            prediction = ensemble(models, item, window)
            extra = {"cached_max_abs_diff": cached_base_diff(name, prediction)} \
                if family == "reference" and setting == BASE else {}
            for label in C.LABEL_SETS:
                result = score_prediction(name, prediction, item, label, selections[label][name])
                rows.append({"family": family, "setting": setting.name, "match": name, "label_set": label,
                             **setting.as_dict(), **stats, **extra, **result})
            log(f"{family} {setting.name} {name} {time.time() - start:.1f}s")
        del models
        torch.cuda.empty_cache()
    return rows


def deltas(rows: list[dict], setting: str, label_set: str, family: str = "reference") -> list[float]:
    """Per-match spell-F1(setting) - spell-F1(target_like), in MATCH_IDS order."""
    def spell(name: str, which: str) -> float:
        (row,) = [r for r in rows if r["family"] == family and r["setting"] == which
                  and r["match"] == name and r["label_set"] == label_set]
        return row["spell_f1"]

    return [spell(name, setting) - spell(name, BASE.name) for name in MATCH_IDS]


# ---------------------------------------------------------------- rechecks
def recheck_tokens() -> dict:
    """S(0) tokens equal exp_2's cached target_like tokens (every array, every match)."""
    out = {}
    for name in MATCH_IDS:
        view, _ = simulate(V.target_like(C1.load_degraded(name, 0)), C1.load_clean(name), BASE,
                           np.array([1]), np.random.default_rng(sim_seed(name)))
        item, cached = token_item(view), P2.load_tokens(name, "target_like", 0)
        out[name] = all(np.array_equal(item[key], cached[key]) for key in item)
    return {"per_match": out, "all_equal": all(out.values())}


def _compare(mine: list[dict], stored: list[dict]) -> dict:
    by_key = {(r["match"], r["label_set"]): r for r in stored}
    diffs = [max(abs(float(r[k]) - float(by_key[(r["match"], r["label_set"])][k])) for k in ROW_KEYS) for r in mine]
    return {"rows": len(mine), "max_abs_diff": float(max(diffs)), "rows_equal": bool(max(diffs) == 0.0)}


def recheck_base(rows: list[dict], family: str = "reference") -> dict:
    """target_like rows through the simulator path equal the stored rows (exp_3 or exp_5)."""
    spec = FAMILIES[family]
    stored = [{**r, "label_set": label} for label in C.LABEL_SETS for r in C1.read_json(spec["result"](label))["rows"]]
    mine = [r for r in rows if r["family"] == family and r["setting"] == BASE.name]
    out = _compare(mine, stored)
    if family == "reference":
        out["max_cached_probability_diff"] = float(max(r["cached_max_abs_diff"] for r in mine))
    return out


def recheck_s2(rows: list[dict]) -> dict:
    """exp_3's S2 coverage view regenerated in memory reproduces exp_3's stored rows and deltas."""
    import pandas as pd

    table = pd.read_csv(C.C3.RESULTS / "coverage_rows.csv")
    table = table[table["variant"] == C.REFERENCE]
    stored = table.to_dict("records")
    mine = [r for r in rows if r["family"] == "reference" and r["setting"] == S2_RECHECK]
    out = _compare(mine, stored)
    base = {(r["match"], r["label_set"]): r["spell_f1"]
            for label in C.LABEL_SETS for r in
            [{**x, "label_set": label} for x in C1.read_json(C.exp3_result(C.REFERENCE, label))["rows"]]}
    for label in C.LABEL_SETS:
        stored_delta = [float(table[(table["match"] == m) & (table["label_set"] == label)]["spell_f1"].iloc[0])
                        - base[(m, label)] for m in MATCH_IDS]
        out[f"delta_{label}"] = {"mine": float(np.mean(deltas(rows, S2_RECHECK, label))),
                                 "stored": float(np.mean(stored_delta))}
    return out
