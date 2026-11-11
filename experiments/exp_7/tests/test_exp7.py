"""Synthetic tests for exp_7 (no BAR-ATH data, no IDSSE fit or scoring)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp7_common as C  # noqa: E402,I001  (environment first)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import exp1_decode as D  # noqa: E402
import exp1_net as N  # noqa: E402
import exp1_tokens as T  # noqa: E402
import exp2_views as V  # noqa: E402
import exp7_aug as A  # noqa: E402
import exp7_decision as DE  # noqa: E402
import exp7_fit as F  # noqa: E402
import exp7_target as TG  # noqa: E402
from exp1_common import JD, MATCH_IDS, match_seed  # noqa: E402

S = C.S6
LENGTHS = np.array([1, 1, 2, 3, 5, 8, 13, 40])
NAME = MATCH_IDS[0]


def synthetic_match(n: int = 2000, seed: int = 3) -> tuple[JD.Match, JD.Match]:
    """(clean, base) where base hides the ball on three long runs."""
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(1000), [0], np.full(n - 1001, 2)].astype("int8")
    players = np.zeros((n, 6, 3), dtype="float32")
    players[:, :, 0] = 50 + np.cumsum(rng.normal(0, 0.1, (n, 6)), axis=0)
    players[:, :, 1] = 34 + np.cumsum(rng.normal(0, 0.1, (n, 6)), axis=0)
    ball = np.stack([50 + np.cumsum(rng.normal(0, 0.3, n)), 34 + np.cumsum(rng.normal(0, 0.3, n))], 1)
    labels = (np.arange(n) // 150 % 3).astype("int8")
    clean = JD.Match("SYN", players, ball.astype("float32"), np.array([0, 0, 0, 1, 1, 1], "int8"), half,
                     np.arange(n), labels, (105.0, 68.0))
    hidden_ball = clean.ball.copy()
    for start, stop in ((100, 400), (600, 900), (1200, 1700)):
        hidden_ball[start:stop] = np.nan
    return clean, replace(clean, ball=hidden_ball)


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a == b).all(axis=1) | (np.isnan(a).all(axis=1) & np.isnan(b).all(axis=1))


# ---------------------------------------------------------------- environment
def test_exp6_imports_leave_exp7_environment_and_numba_cache():
    import numba

    assert os.environ["NUMBA_CACHE_DIR"] == C.ENV["NUMBA_CACHE_DIR"]
    assert os.environ["MPLCONFIGDIR"] == C.ENV["MPLCONFIGDIR"]
    assert Path(numba.config.CACHE_DIR) == ROOT / "cache" / "numba"
    C.import_exp6("exp6_sim")
    assert os.environ["NUMBA_CACHE_DIR"] == C.ENV["NUMBA_CACHE_DIR"]


# ---------------------------------------------------------------- augmentation
def test_augmentation_changes_only_hidden_frames_and_flags_exactly_them():
    clean, base = synthetic_match()
    view, info = A.augment(base, clean, A.train_seed(NAME, 1), LENGTHS)
    hidden = S.hidden_frames(base, clean.ball)
    changed = ~_same(view.ball, base.ball)
    assert changed.any() and not (changed & ~hidden).any()
    mask = A.recovered_mask(base, view)
    assert np.array_equal(mask, changed)  # synthetic players are complete: every revealed frame has a ball
    assert mask.sum() == info["revealed_frames"]
    assert np.array_equal(view.labels, base.labels) and np.array_equal(view.players, base.players)
    for key, (low, high) in A.TRAIN_RANGES.items():
        assert low <= info[key] <= high
    item = A.token_item(view, mask, labels=True)
    assert np.array_equal(item["glob"][:, -1].astype(bool), mask[item["indices"]])
    assert item["glob"].shape[1] == len(T.GLOBAL_NAMES) + 1


def test_augmentation_seed_is_fixed():
    clean, base = synthetic_match()
    first, info1 = A.augment(base, clean, 11, LENGTHS)
    second, info2 = A.augment(base, clean, 11, LENGTHS)
    third, _ = A.augment(base, clean, 12, LENGTHS)
    assert np.array_equal(first.ball, second.ball, equal_nan=True) and info1 == info2
    assert not np.array_equal(first.ball, third.ball, equal_nan=True)


def test_training_seeds_are_disjoint_from_exp6_and_earlier_seeds():
    train = {A.train_seed(n, r) for n in MATCH_IDS for r in A.SIM_ROUNDS}
    exp6 = {S.sim_seed(n) for n in MATCH_IDS}
    earlier = {V.aug_seed(n, r) for n in MATCH_IDS for r in (0, 1, 2)} | \
              {match_seed(n, r) for n in MATCH_IDS for r in (0, 1, 2)}
    assert len(train) == len(MATCH_IDS) * len(A.SIM_ROUNDS)
    assert not train & exp6 and not train & earlier


def test_zero_flag_on_unsimulated_items():
    clean, base = synthetic_match()
    item = A.token_item(base, A.recovered_mask(base, base), labels=False)
    assert not item["glob"][:, -1].any()
    assert np.array_equal(A.without_flag(item)["glob"], T.build_tokens(base).glob)


# ---------------------------------------------------------------- extra global
def test_flag_globals_restores_exp1_net_and_builds_the_right_width():
    saved = (N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP)
    with A.flag_globals():
        assert N.GLOBAL_NAMES[-1] == A.FLAG_NAME
        model = N.SetSeq(8, 8)
    assert (N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP) == saved
    assert model.frame[0].in_features == 5 * 8 + len(T.GLOBAL_NAMES) + 1
    with pytest.raises(ValueError):
        with A.flag_globals():
            raise ValueError("boom")
    assert (N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP) == saved


def test_mirror_leaves_the_flag_unchanged():
    rng = np.random.default_rng(0)
    tokens = torch.from_numpy(rng.normal(size=(2, 5, 4, len(T.TOKEN_NAMES))).astype("float32")).to(N.DEVICE)
    glob = torch.from_numpy(rng.normal(size=(2, 5, len(T.GLOBAL_NAMES) + 1)).astype("float32")).to(N.DEVICE)
    state = torch.zeros((2, 5), dtype=torch.int64, device=N.DEVICE)
    flip = torch.tensor([True, False], device=N.DEVICE)
    with A.flag_globals():
        _, mirrored, _ = N._mirror_batch(tokens, glob, state, flip)
    assert torch.equal(mirrored[..., -1], glob[..., -1])
    _, expected, _ = T.mirror(np.zeros((5, 4, len(T.TOKEN_NAMES))), glob[0, :, :-1].cpu().numpy(), np.zeros(5, "int8"))
    assert np.allclose(mirrored[0, :, :-1].cpu().numpy(), expected)
    assert torch.equal(mirrored[1], glob[1])


# ---------------------------------------------------------------- evaluation views
def test_measured_and_stress_views_equal_exp6():
    clean, base = synthetic_match()
    settings = A.eval_settings()
    for view_name in ("measured", "stress", "clean25"):
        mine, _ = A.eval_view(NAME, view_name, LENGTHS, base, clean)
        theirs, _ = S.simulate(base, clean, settings[view_name], LENGTHS, np.random.default_rng(S.sim_seed(NAME)))
        assert np.array_equal(mine.ball, theirs.ball, equal_nan=True)
    assert settings["measured"] == S.measured(C.measured_inputs())
    assert settings["stress"] == S.stress(C.measured_inputs())


def test_ar1_with_rho_zero_is_exp6_simulate_and_keeps_the_measured_subruns():
    clean, base = synthetic_match()
    setting = A.eval_settings()["measured"]
    iid, stats = S.simulate(base, clean, setting, LENGTHS, np.random.default_rng(5))
    ar0, _ = A.simulate_ar1(base, clean, setting, LENGTHS, np.random.default_rng(5), 0.0)
    assert np.array_equal(iid.ball, ar0.ball, equal_nan=True)
    ar9, stats9 = A.simulate_ar1(base, clean, setting, LENGTHS, np.random.default_rng(5), A.AR1_RHO)
    assert np.array_equal(A.recovered_mask(base, ar9), A.recovered_mask(base, iid))
    assert stats9["fp_frames"] == stats["fp_frames"] and stats9["subruns"] == stats["subruns"]


def test_ar1_noise_is_stationary_sigma_and_correlated():
    z = np.random.default_rng(1).normal(size=(200_000, 2))
    e = A.ar1_noise(z, 0.5, 0.9)
    assert np.allclose(e.std(axis=0), 0.5, rtol=0.03)
    lag1 = np.mean([np.corrcoef(e[1:, k], e[:-1, k])[0, 1] for k in (0, 1)])
    assert abs(lag1 - 0.9) < 0.01


# ---------------------------------------------------------------- decision rule
GOOD = {"gain": [0.02, 0.03, 0.01, 0.02, 0.025, 0.015, 0.02], "no_harm": [0.0] * 7, "stress": [0.0] * 7}
BAD = {"gain": [-0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0], "no_harm": [-0.006] * 7, "stress": [-0.006] * 7}


def test_every_clause_on_every_label_set_can_fail_h1():
    assert DE.decide(DE.clause_table({label: GOOD for label in C.LABEL_SETS}))["h1_passed"]
    for label in C.LABEL_SETS:
        for clause in GOOD:
            deltas = {ls: dict(GOOD) for ls in C.LABEL_SETS}
            deltas[label] = {**GOOD, clause: BAD[clause]}
            decision = DE.decide(DE.clause_table(deltas))
            assert not decision["h1_passed"] and decision["failed"] == [f"{clause}:{label}"]


def test_files_are_written_only_on_a_pass_and_the_guard_blocks(tmp_path):
    codes = np.full(TG.X1.N_FRAMES, 2, dtype="int8")
    with pytest.raises(RuntimeError):
        TG.write_candidate(tmp_path / "a.csv", codes, {"h1_passed": False})
    passed = {"h1_passed": True}
    blocked = TG.guarded_file("ctrl", codes, {"dead_share": 0.5}, tmp_path / "b.csv", passed)
    assert not blocked["written"] and not (tmp_path / "b.csv").exists()
    written = TG.guarded_file("ctrl", codes, {"dead_share": 0.41}, tmp_path / "c.csv", passed)
    assert written["written"] and (tmp_path / "c.csv").exists() and len(written["sha256"]) == 64
    assert DE.plausible(0.407 + 0.029) and not DE.plausible(0.407 - 0.031)


# ---------------------------------------------------------------- fits
def test_fit_false_raises_and_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PRED_DIR", tmp_path / "predictions")
    with pytest.raises(RuntimeError):
        F.ensure_predictions("exp7_robust", fit=False)
    assert not (tmp_path / "predictions").exists()


def test_fingerprint_changes_when_a_hashed_module_changes(tmp_path, monkeypatch):
    shutil.copy(ROOT / "exp7_aug.py", tmp_path / "exp7_aug.py")
    monkeypatch.setattr(F, "FIT_CODE", {**F.FIT_CODE, "exp7_aug.py": tmp_path})
    before = F.fingerprint("exp7_robust")
    assert before == F.fingerprint("exp7_robust")
    with (tmp_path / "exp7_aug.py").open("a") as handle:
        handle.write("\n# changed\n")
    assert F.fingerprint("exp7_robust") != before
    assert len({F.fingerprint(arm) for arm in F.ARMS}) == len(F.ARMS)


# ---------------------------------------------------------------- isolation
def test_tiny_fit_and_evaluation_leave_exp0_to_exp6_and_data_unchanged(tmp_path, monkeypatch):
    before = C.write_manifest()
    for name in ("CACHE", "TOKEN_DIR", "PRED_DIR", "TARGET_MODEL_DIR", "REBUILD_DIR", "RESULTS", "PREDICTIONS_OUT"):
        monkeypatch.setattr(C, name, tmp_path / name.lower())
    clean, base = synthetic_match()
    view, _ = A.augment(base, clean, 3, LENGTHS)
    item = A.token_item(view, A.recovered_mask(base, view), labels=True)
    A._save(C.TOKEN_DIR / "syn.npz", item)
    config = replace(N.TrainConfig(), window=20, epochs=1, batch=4, width=8, hidden=8)
    with A.flag_globals():
        model = N.train([A._load(C.TOKEN_DIR / "syn.npz")], config)
    probability, boundary = N.predict(model, item, config.window)
    evaluation = {"probability": probability, "boundary": boundary, "half": item["half"],
                  "indices": item["indices"], "state": item["state"], "truth": view.labels, "full_half": view.half}
    selected, _ = D.select_config([evaluation], (0.0,))
    assert 0.0 <= D.score_item(evaluation, selected)["spell_f1"] <= 1.0
    A.eval_view(NAME, "ar1", LENGTHS, base, clean)
    assert C.write_manifest() == before
