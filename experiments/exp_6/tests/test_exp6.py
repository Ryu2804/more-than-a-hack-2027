"""Synthetic tests for exp_6 (no BAR-ATH data, no IDSSE scoring)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp6_common as C  # noqa: E402  (environment first)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import exp6_data as DA  # noqa: E402
import exp6_decision as DE  # noqa: E402
import exp6_detector as DT  # noqa: E402
import exp6_linker as LK  # noqa: E402
import exp6_recover as R  # noqa: E402
import exp6_sim as S  # noqa: E402
import exp6_target as TG  # noqa: E402
from exp1_common import JD  # noqa: E402

TINY = DT.DetectorConfig(channels=(8, 8, 16, 16), crop=32, central=16, sigma_px=1.5, pos_weight=20.0,
                         near_player_min_px=10.0, passes=6, buffer=32, batch=8, lr=3e-3, lr_min=1e-4,
                         seed=7, infer_margin_px=0, top_k=5, nms=7)
N_VIDEO, HEIGHT, WIDTH = 80, 64, 96


# ---------------------------------------------------------------- fixtures
def synthetic_video(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noisy background with a bright 3x3 dot moving on a straight line (BGR uint8)."""
    rng = np.random.default_rng(seed)
    video = rng.integers(40, 80, (N_VIDEO, HEIGHT, WIDTH, 3), dtype=np.uint8)
    centres = np.zeros((N_VIDEO, 2))
    for t in range(N_VIDEO):
        cx, cy = int(round(20 + 0.6 * t)), int(round(20 + 0.3 * t))
        video[t, cy - 1:cy + 2, cx - 1:cx + 2] = 250
        centres[t] = (cx, cy)
    return video, centres


def tiny_train_data(centres: np.ndarray) -> DT.TrainData:
    targets = np.zeros(N_VIDEO, dtype=bool)
    targets[1:70] = True
    label_uv = np.where(targets[:, None], centres, np.nan)
    players = np.tile(np.array([[85.0, 8.0], [np.nan, np.nan]]), (int(targets.sum()), 1, 1))
    return DT.TrainData(targets=targets, label_uv=label_uv, bbox=np.tile([0, 0, WIDTH, HEIGHT], (N_VIDEO, 1)),
                        players_uv=players, image_size=(WIDTH, HEIGHT))


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> dict:
    video, centres = synthetic_video()
    data = tiny_train_data(centres)
    folders = [tmp_path_factory.mktemp(f"det{k}") for k in range(2)]
    models = [DT.train(DA.ArraySource(video), data, TINY, folder, log=lambda *_: None) for folder in folders]
    return {"video": video, "centres": centres, "data": data, "models": models, "folders": folders}


def synthetic_match(n: int = 2000, seed: int = 3) -> tuple[JD.Match, JD.Match]:
    """(clean, base) where base hides the ball on three long runs."""
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(1000), [0], np.full(n - 1001, 2)].astype("int8")
    players = np.zeros((n, 6, 3), dtype="float32")
    players[:, :, 0] = 50 + np.cumsum(rng.normal(0, 0.1, (n, 6)), axis=0)
    players[:, :, 1] = 34 + np.cumsum(rng.normal(0, 0.1, (n, 6)), axis=0)
    ball = np.stack([50 + np.cumsum(rng.normal(0, 0.3, n)), 34 + np.cumsum(rng.normal(0, 0.3, n))], 1)
    clean = JD.Match("SYN", players, ball.astype("float32"), np.array([0, 0, 0, 1, 1, 1], "int8"), half,
                     np.arange(n), np.zeros(n, "int8"), (105.0, 68.0))
    hidden_ball = clean.ball.copy()
    for start, stop in ((100, 400), (600, 900), (1200, 1700)):
        hidden_ball[start:stop] = np.nan
    return clean, replace(clean, ball=hidden_ball)


# ---------------------------------------------------------------- splits and labels
def test_splits_follow_k_mod_5_and_are_disjoint():
    half = np.r_[np.zeros(5), np.ones(200), np.zeros(5), np.full(200, 2)].astype("int8")
    visible = np.zeros(len(half), dtype=bool)
    for start in range(5, 410, 17):
        visible[start:start + 9] = True
    split = DA.split_frames(visible, half)
    assert set(np.unique(split[half == 0])) == {DA.UNGRADED}
    assert np.all(split[(half > 0) & ~visible] == DA.HIDDEN)
    for period in (1, 2):
        index = np.flatnonzero(half == period)
        for k, (a, b) in enumerate(JD.runs(visible[index])):
            codes = set(split[index[a:b]])
            expected = DA.TRAIN if k % 5 in (0, 1, 2) else DA.VAL if k % 5 == 3 else DA.TEST
            assert codes == {expected}


def test_hidden_val_test_frames_never_reach_the_training_sampler():
    n = 300
    half = np.r_[np.zeros(10), np.ones(n - 10)].astype("int8")
    visible = (np.arange(n) // 13) % 2 == 0
    split = DA.split_frames(visible, half)
    keep = visible & (half > 0)
    inputs = DA.TargetInputs(ball=np.where(visible[:, None], 50.0, np.nan), players=np.full((n, 3, 2), 40.0),
                             half=half, matrices=np.tile(np.eye(3), (n, 1, 1)), split=split,
                             label_uv=np.where(keep[:, None], 100.0, np.nan), keep=keep,
                             bbox=np.tile([0, 0, 1920, 1080], (n, 1)))
    targets = R.train_data(inputs).targets
    assert targets.any()
    assert np.all(split[targets] == DA.TRAIN)
    frames = DA.ArraySource(np.zeros((n, 4, 4, 3), np.uint8)).frames(DA.dilate(targets))
    yielded = [t for t, _ in DA.triplets(frames, targets)]
    assert yielded and all(split[t] == DA.TRAIN for t in yielded)
    assert len(yielded) == int(targets[1:-1].sum())


def test_projection_round_trip_and_starter_formula():
    rng = np.random.default_rng(1)
    base = np.array([[1800.0, -500.0, 50.0], [2.0, 170.0, 250.0], [-0.02, -0.5, 1.0]])
    matrices = base[None] * (1 + 0.01 * rng.normal(size=(50, 3, 3)))
    x, y = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50)
    u, v = DA.project(matrices, x, y)
    back_x, back_y = DA.project(np.linalg.inv(matrices), u, v)
    assert np.abs(back_x - x).max() < 1e-6 and np.abs(back_y - y).max() < 1e-6
    starter = DA._target_repo_module("load_homography")
    su, sv = starter.project(matrices[0], x[0], y[0])
    assert abs(su - u[0]) < 1e-9 and abs(sv - v[0]) < 1e-9
    metres = rng.uniform(0, 60, (50, 2))
    pu, pv = DA.metres_to_pixels(matrices, metres)
    assert np.abs(DA.pixels_to_metres(np.linalg.inv(matrices), pu, pv) - metres).max() < 1e-6


def test_candidate_pixels_map_back_to_metres_with_their_own_frame_matrix():
    """The (n, 3, 3) x (n, k) broadcast used for 02's quality numbers (post-hoc check)."""
    rng = np.random.default_rng(2)
    base = np.array([[1800.0, -500.0, 50.0], [2.0, 170.0, 250.0], [-0.02, -0.5, 1.0]])
    matrices = base[None] * (1 + 0.01 * rng.normal(size=(20, 3, 3)))
    metres = rng.uniform([0.0, 0.0], DA.PITCH, (20, 5, 2))
    u, v = DA.metres_to_pixels(matrices, metres)
    back = DA.pixels_to_metres(np.linalg.inv(matrices), u, v)
    assert back.shape == (20, 5, 2) and np.abs(back - metres).max() < 1e-6
    assert np.abs(DA.pixels_to_metres(np.linalg.inv(matrices[::-1]), u, v) - metres).max() > 1e-3


# ---------------------------------------------------------------- detector
def test_detector_training_is_bit_identical(trained):
    first, second = (m.state_dict() for m in trained["models"])
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_detector_finds_the_dot(trained):
    video, centres, model = trained["video"], trained["centres"], trained["models"][0]
    for t in (74, 76):
        frames = tuple(DT.to_device(video[i]) for i in (t - 1, t, t + 1))
        uv, _ = DT.detect(model, frames, np.array([0, 0, WIDTH, HEIGHT]), TINY)
        assert np.hypot(*(uv[0] - centres[t])) <= 2.0


def test_resume_from_checkpoint_is_bit_identical(trained, tmp_path):
    shutil.copy(trained["folders"][0] / "pass_0.pt", tmp_path / "pass_0.pt")
    resumed = DT.train(DA.ArraySource(trained["video"]), trained["data"], TINY, tmp_path, log=lambda *_: None)
    reference = trained["models"][0].state_dict()
    assert all(torch.equal(resumed.state_dict()[k], reference[k]) for k in reference)


# ---------------------------------------------------------------- linker
def _track(points: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    xy, score = np.full((n, 5, 2), np.nan), np.full((n, 5), np.nan)
    xy[:, 0], score[:, 0] = points, scores
    return np.arange(n) + 100, xy, score


HALF = np.ones(1000, dtype="int8")
CFG = LK.LinkerConfig()


def test_speed_jump_above_40_mps_is_never_linked():
    x = np.r_[0.5 * np.arange(10), 30 + 0.5 * np.arange(10)]
    frames, xy, score = _track(np.stack([x, np.zeros(20)], 1), np.full(20, 0.99))
    xy[:, 1] = xy[:, 0] + [2.5, 0.0]  # a second candidate always 2.5 m away
    score[:, 1] = 0.999
    linked = LK.link(frames, xy, score, HALF, CFG)
    on = linked["state"] < linked["k"]
    assert on.any()
    steps = np.linalg.norm(np.diff(linked["position"], axis=0), axis=1)
    assert np.all(steps[on[1:] & on[:-1]] <= CFG.max_step_m + 1e-12)


def test_short_isolated_detections_are_not_accepted():
    points = np.stack([0.3 * np.arange(10), np.zeros(10)], 1)
    scores = np.r_[np.full(3, 0.99), np.full(7, np.nan)]
    frames, xy, score = _track(points, scores)
    linked = LK.link(frames, xy, score, HALF, CFG)
    assert not LK.accept(linked, 0.1, CFG).any()


def test_acceptance_threshold_is_respected():
    frames, xy, score = _track(np.stack([0.3 * np.arange(10), np.zeros(10)], 1), np.full(10, 0.6))
    linked = LK.link(frames, xy, score, HALF, CFG)
    assert LK.accept(linked, 0.5, CFG).all()
    assert not LK.accept(linked, 0.7, CFG).any()


def test_segments_break_at_gaps_and_halves():
    frames = np.array([10, 11, 12, 20, 21, 500, 501])
    half = np.zeros(1000, dtype="int8")
    half[:400], half[400:] = 1, 2
    assert LK.segments(frames, half) == [(0, 3), (3, 5), (5, 7)]


# ---------------------------------------------------------------- simulator
LENGTHS = np.array([7, 13, 25, 40])


def test_simulator_zero_coverage_is_the_base_view():
    clean, base = synthetic_match()
    view, stats = S.simulate(base, clean, S.Setting("zero", 0.0, 0.5, 0.5), LENGTHS, np.random.default_rng(0))
    assert stats["revealed_frames"] == 0
    assert np.array_equal(view.ball, base.ball, equal_nan=True) and view.ball.dtype == base.ball.dtype


def test_simulator_revealed_share_and_visible_frames():
    clean, base = synthetic_match()
    hidden = S.hidden_frames(base, clean.ball)
    for coverage in (0.12, 0.3, 1.0):
        view, stats = S.simulate(base, clean, S.Setting("c", coverage, 0.4, 0.0), LENGTHS, np.random.default_rng(1))
        newly = np.isfinite(view.ball).all(1) & ~np.isfinite(base.ball).all(1)
        assert abs(int(newly.sum()) - coverage * hidden.sum()) <= LENGTHS.max()
        assert int(newly.sum()) == stats["target_frames"] == stats["revealed_frames"]
        assert not np.any(newly & ~hidden)
        visible = np.isfinite(base.ball).all(1)
        assert np.array_equal(view.ball[visible], base.ball[visible])


def test_simulator_noise_is_sigma_metres_per_axis():
    """sigma is per-axis metres on revealed frames (post-hoc check of the measured-setting units)."""
    clean, base = synthetic_match()
    view, _ = S.simulate(base, clean, S.Setting("n", 1.0, 0.5, 0.0), LENGTHS, np.random.default_rng(5))
    newly = np.isfinite(view.ball).all(1) & ~np.isfinite(base.ball).all(1)
    residual = view.ball[newly].astype("float64") - clean.ball[newly].astype("float64")
    assert newly.sum() == S.hidden_frames(base, clean.ball).sum()
    assert abs(float(np.sqrt(np.mean(residual ** 2))) - 0.5) < 0.05


def test_false_positive_runs_follow_a_player_and_seed_is_fixed():
    clean, base = synthetic_match()
    setting = S.Setting("fp", 0.5, 0.0, 1.0)
    view, stats = S.simulate(base, clean, setting, LENGTHS, np.random.default_rng(2))
    again, _ = S.simulate(base, clean, setting, LENGTHS, np.random.default_rng(2))
    assert np.array_equal(view.ball, again.ball, equal_nan=True)
    assert stats["fp_frames"] == stats["revealed_frames"] > 0
    newly = np.flatnonzero(np.isfinite(view.ball).all(1) & ~np.isfinite(base.ball).all(1))
    nearest = np.linalg.norm(clean.players[newly, :, :2] - view.ball[newly][:, None], axis=-1).min(axis=1)
    assert np.all(nearest < 5.0)


# ---------------------------------------------------------------- decision rule
def _ci(provider: float, ctrl: float) -> dict:
    return {"provider": {"ci_low": provider, "mean_delta": 0.01, "pass": provider > 0},
            "ctrl": {"ci_low": ctrl, "mean_delta": 0.01, "pass": ctrl > 0}}


def _stress(provider: float, ctrl: float) -> dict:
    return {"provider": {"mean_delta": provider, "pass": provider >= -0.005},
            "ctrl": {"mean_delta": ctrl, "pass": ctrl >= -0.005}}


GOOD_GATES = DE.gate_table(0.99, 0.95, 0.6, 0.3)


def test_decision_rule_all_clauses():
    assert DE.decide(GOOD_GATES, _ci(0.002, 0.001), _stress(0.0, -0.004))["h1_passed"]
    gate_fail = DE.decide(DE.gate_table(0.99, 0.85, 0.6, 0.3), _ci(0.002, 0.001), _stress(0.0, 0.0))
    assert not gate_fail["h1_passed"] and gate_fail["failed"] == ["G2"]
    assert not DE.decide(DE.gate_table(0.99, float("nan"), 0.0, 0.0), _ci(0.1, 0.1), _stress(0, 0))["h1_passed"]
    ci_fail = DE.decide(GOOD_GATES, _ci(0.002, -0.001), _stress(0.0, 0.0))
    assert not ci_fail["h1_passed"] and ci_fail["failed"] == ["ci_clause:ctrl"]
    stress_fail = DE.decide(GOOD_GATES, _ci(0.002, 0.001), _stress(-0.006, 0.0))
    assert not stress_fail["h1_passed"] and stress_fail["failed"] == ["stress_clause:provider"]


def test_ci_and_stress_clause_thresholds():
    assert not DE.ci_clause({"provider": [0.0] * 7, "ctrl": [0.01] * 7})["provider"]["pass"]
    assert DE.stress_clause({"provider": [-0.005] * 7, "ctrl": [-0.004] * 7})["provider"]["pass"]
    assert not DE.stress_clause({"provider": [-0.006] * 7, "ctrl": [0.0] * 7})["provider"]["pass"]


def test_files_are_written_only_on_a_pass(tmp_path):
    path = tmp_path / "candidate.csv"
    codes = np.full(DA.N_FRAMES, 2, dtype="int8")
    with pytest.raises(RuntimeError):
        TG.write_candidate(path, codes, {"h1_passed": False})
    with pytest.raises(RuntimeError):
        TG.write_candidate(path, codes, {})
    assert not path.exists()
    TG.write_candidate(path, codes, {"h1_passed": True})
    assert path.exists()


# ---------------------------------------------------------------- isolation
def test_tiny_end_to_end_leaves_exp0_to_exp5_and_data_unchanged(tmp_path, monkeypatch, trained):
    before = C.write_manifest()
    for name in ("CACHE", "DETECTOR_DIR", "INFER_DIR", "REBUILD_DIR", "RESULTS", "PREDICTIONS_OUT"):
        monkeypatch.setattr(C, name, tmp_path / name.lower())
    video, centres = synthetic_video(seed=5)
    data = tiny_train_data(centres)
    model = DT.train(DA.ArraySource(video), data, replace(TINY, passes=1), C.DETECTOR_DIR, log=lambda *_: None)
    mask = np.zeros(N_VIDEO, dtype=bool)
    mask[70:78] = True
    raw = DT.infer(model, DA.ArraySource(video), mask, np.tile([0, 0, WIDTH, HEIGHT], (N_VIDEO, 1)), TINY,
                   C.INFER_DIR, log=lambda *_: None, chunk=3)
    linked = LK.link(raw["frame"], raw["uv"] / 10.0, 1 / (1 + np.exp(-raw["logit"])),
                     np.ones(N_VIDEO, "int8"), CFG)
    assert linked["state"].shape == (8,)
    clean, base = synthetic_match()
    S.simulate(base, clean, S.Setting("c", 0.3, 0.4, 0.2), LENGTHS, np.random.default_rng(4))
    DE.decide(GOOD_GATES, _ci(0.002, 0.001), _stress(0.0, 0.0))
    assert C.write_manifest() == before
