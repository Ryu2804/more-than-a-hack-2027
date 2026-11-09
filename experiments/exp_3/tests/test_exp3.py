"""Synthetic tests for exp_3: label relabelings, event sync, replay guard, coverage view."""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import exp3_common as C  # noqa: E402,F401  (sets up exp_1 / exp_2 imports)
import exp3_coverage as V3  # noqa: E402
import exp3_fusion as F  # noqa: E402
import exp3_labels as L  # noqa: E402
from exp1_common import Match  # noqa: E402

H, A, D = 0, 1, 2
FPS = 25
NO_EVENTS = (np.zeros(0, "int64"), np.zeros(0, "int64"))


def build(*segments, half_value=1):
    """Labels from (code, seconds) segments in one half, plus its half array."""
    labels = np.concatenate([np.full(int(round(s * FPS)), c, dtype="int8") for c, s in segments])
    return labels, np.full(len(labels), half_value, dtype="int8")


def two_halves(first, second):
    """Two halves separated by one unlabelled frame with half 0 (as the judge loader does)."""
    l1, h1 = build(*first, half_value=1)
    l2, h2 = build(*second, half_value=2)
    return (np.concatenate([l1, [3], l2]).astype("int8"),
            np.concatenate([h1, [0], h2]).astype("int8"), len(l1) + 1)


# ---------------------------------------------------------------- L_ctrl
def test_short_uncontrolled_run_joins_previous_team():
    labels, half = build((H, 5), (A, 1), (H, 5))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert (out == H).all()


def test_rule_b_long_run_is_controlled():
    labels, half = build((H, 5), (A, 2), (H, 5))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert np.array_equal(out, labels)


def test_rule_a_after_dead_ball_and_at_half_start():
    labels, half = build((A, 0.4), (H, 5), (D, 3), (A, 0.4), (H, 5))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert np.array_equal(out, labels)  # both 0.4 s A runs start the half / follow D


def test_rule_c_event_keeps_short_run_and_respects_team_and_margin():
    labels, half = build((H, 5), (A, 1), (H, 5))
    start, stop = 125, 150
    inside = L.l_ctrl(labels, half, np.array([start + 3]), np.array([A]))
    assert np.array_equal(inside, labels)
    wrong_team = L.l_ctrl(labels, half, np.array([start + 3]), np.array([H]))
    assert (wrong_team == H).all()
    in_margin = L.l_ctrl(labels, half, np.array([stop - 1 + 12]), np.array([A]))
    assert np.array_equal(in_margin, labels)
    beyond = L.l_ctrl(labels, half, np.array([stop - 1 + 13]), np.array([A]))
    assert (beyond == H).all()
    ignored = L.l_ctrl(labels, half, np.array([start + 3]), np.array([A]), use_events=False)
    assert (ignored == H).all()


def test_touch_sending_ball_out_stays_with_previous_team():
    labels, half = build((H, 5), (A, 0.6), (D, 4))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert np.array_equal(out, np.r_[np.full(140, H), np.full(100, D)])


def test_chain_of_short_runs_merges_left_to_right():
    labels, half = build((H, 5), (A, 0.6), (H, 0.6), (A, 0.6), (H, 3))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert (out == H).all()


def test_merge_direction_is_previous_team_not_next():
    labels, half = build((A, 5), (H, 1), (D, 2))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert (out[:150] == A).all() and (out[150:] == D).all()


def test_dead_frames_untouched():
    labels, half = build((H, 5), (A, 1), (D, 2), (H, 1), (A, 0.5), (H, 3))
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert np.array_equal(out == D, labels == D)
    assert not np.shares_memory(out, labels)


def test_never_merges_across_halves():
    labels, half, cut = two_halves([(H, 5)], [(A, 0.4), (H, 5)])
    out = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert out[cut:cut + 10].tolist() == [A] * 10  # first run of half 2 is controlled (rule a)
    assert out[cut - 1] == 3
    event_in_half1 = L.l_ctrl(labels, half, np.array([cut - 2]), np.array([A]))
    assert np.array_equal(event_in_half1, labels)


def test_ctrl_without_events_equals_track_2s():
    rng = np.random.default_rng(0)
    codes = rng.choice([H, A, D], size=200, p=[0.45, 0.45, 0.1])
    seconds = rng.choice([0.2, 0.6, 1.0, 1.6, 2.0, 2.4, 5.0], size=200)
    labels, half = build(*zip(codes, seconds))
    ctrl = L.l_ctrl(labels, half, *NO_EVENTS, use_events=False)
    assert np.array_equal(ctrl, L.l_track(labels, half, 2.0))


# ---------------------------------------------------------------- L_track
def test_track_tau_threshold_and_dead_follow():
    labels, half = build((H, 5), (A, 0.96), (D, 2), (A, 1.0), (H, 0.5), (A, 3))
    out = L.l_track(labels, half, 1.0)
    assert (out[125:149] == H).all()  # 24 frames < 25 -> merged, followed by D
    assert (out[199:224] == A).all()  # 25 frames: not shorter than tau
    assert (out[224:236] == A).all()  # H 0.5 s between A runs -> A
    assert np.array_equal(out == D, labels == D)


# ---------------------------------------------------------------- events
def test_event_frame_sync_from_kickoff():
    kickoff = datetime(2023, 5, 27, 15, 30, 12, 230000)
    assert L.event_frame(kickoff, kickoff, 10000) == 10000
    assert L.event_frame(kickoff + timedelta(seconds=2.0), kickoff, 10000) == 10050
    assert L.event_frame(kickoff + timedelta(seconds=0.019), kickoff, 100000) == 100000
    assert L.event_frame(kickoff + timedelta(seconds=0.021), kickoff, 100000) == 100001


def test_controlled_event_types():
    def element(xml):
        return etree.fromstring(xml)

    assert L.controlled_team(element('<Play Team="T1" Evaluation="successfullyCompleted"/>')) == "T1"
    assert L.controlled_team(element('<Play Team="T1" Evaluation="unsuccessful"/>')) is None
    assert L.controlled_team(element('<ShotAtGoal Team="T2"/>')) == "T2"
    assert L.controlled_team(element('<BallClaiming Team="T2" Type="InterceptedBall"/>')) == "T2"
    assert L.controlled_team(element('<BallClaiming Team="T2" Type="Block"/>')) is None
    won = '<TacklingGame WinnerTeam="T1" Winner="P1" WinnerResult="ballClaimed"/>'
    assert L.controlled_team(element(won)) == "T1"
    touch = '<TacklingGame WinnerTeam="T1" WinnerResult="ballcontactSucceeded"/>'
    assert L.controlled_team(element(touch)) is None
    assert L.controlled_team(element('<OtherBallAction Team="T1"/>')) is None
    assert L.event_player(element(won)) == "P1"


def test_snap_moves_event_to_closest_approach_and_drops_unlocated():
    n = 400
    half = np.ones(n, dtype="int8")
    ball = np.zeros((n, 2), dtype="float32")
    players = np.full((n, 2, 3), 50.0, dtype="float32")
    players[200, 0, :2] = 0.5  # slot 0 touches the ball at frame 200
    events = L.MatchEvents(index=np.array([260, 260]), team=np.array([0, 1]), kind=np.array(["a", "b"]),
                           slot=np.array([0, 1]), calculated=np.array([-1, -1]), dropped=0)
    snapped = L.snap_events(events, players, ball, half)
    assert snapped.index.tolist() == [200]
    assert snapped.team.tolist() == [0]
    assert snapped.dropped == 1


def test_snap_window_stays_in_half():
    half = np.r_[np.ones(100), np.full(100, 2)].astype("int8")
    ball = np.zeros((200, 2), dtype="float32")
    players = np.full((200, 1, 3), 50.0, dtype="float32")
    players[90, 0, :2] = 0.0  # close approach in half 1 only
    events = L.MatchEvents(np.array([120]), np.array([0]), np.array(["a"]), np.array([0]),
                           np.array([-1]), 0)
    assert len(L.snap_events(events, players, ball, half).index) == 0


def test_sync_quality_rule():
    assert L.sync_quality(np.array([0, 1, -2, 3, 4] * 4))["accepted"]
    assert not L.sync_quality(np.r_[np.zeros(80), np.full(20, 60)])["accepted"]


# ---------------------------------------------------------------- replay guard
def test_replay_guard_rejects_unreachable_and_one_sided_frames():
    n = 100
    half = np.ones(n, dtype="int8")
    tactical = np.full((n, 2), np.nan, dtype="float32")
    tactical[10] = (0.0, 0.0)
    tactical[60] = (10.0, 0.0)
    candidate = tactical.copy()
    candidate[20] = (5.0, 0.0)  # reachable from both sides
    candidate[21] = (500.0, 0.0)  # replay: too far
    candidate[5] = (0.0, 0.0)  # no tactical observation before
    candidate[80] = (10.0, 0.0)  # no tactical observation after
    recovered = np.isnan(tactical).any(axis=1) & np.isfinite(candidate).all(axis=1)
    keep = F.replay_guard(tactical, candidate, recovered, half)
    assert np.flatnonzero(keep).tolist() == [20]


def test_replay_guard_speed_bound_is_per_side():
    half = np.ones(50, dtype="int8")
    tactical = np.full((50, 2), np.nan, dtype="float32")
    tactical[0], tactical[49] = (0.0, 0.0), (0.0, 0.0)
    candidate = tactical.copy()
    candidate[1] = (1.5, 0.0)  # 1 frame = 0.04 s -> 1.6 m reachable at 40 m/s
    candidate[2] = (3.3, 0.0)  # 2 frames -> 3.2 m reachable: rejected
    recovered = np.isnan(tactical).any(axis=1) & np.isfinite(candidate).all(axis=1)
    assert np.flatnonzero(F.replay_guard(tactical, candidate, recovered, half)).tolist() == [1]


def test_replay_guard_does_not_cross_halves():
    half = np.r_[np.ones(20), np.full(20, 2)].astype("int8")
    tactical = np.full((40, 2), np.nan, dtype="float32")
    tactical[5] = (0.0, 0.0)
    tactical[30] = (0.0, 0.0)
    candidate = tactical.copy()
    candidate[15] = (0.0, 0.0)  # half 1: no observation after it inside half 1
    recovered = np.isnan(tactical).any(axis=1) & np.isfinite(candidate).all(axis=1)
    assert not F.replay_guard(tactical, candidate, recovered, half).any()


def test_fusion_pin_verifies():
    assert F.verify_pin()["function"] == "fusion_candidate"


# ---------------------------------------------------------------- coverage view
def _toy_match(n=2000):
    rng = np.random.default_rng(3)
    half = np.r_[np.ones(n // 2), np.full(n // 2, 2)].astype("int8")
    ball = rng.uniform(10, 50, size=(n, 2)).astype("float32")
    for start in (100, 700, 1300):
        ball[start:start + 250] = np.nan
    players = rng.uniform(0, 60, size=(n, 4, 3)).astype("float32")
    players[..., 2] = 3.0  # fast play: target_like hides nothing slow
    return Match("toy", players, ball, np.array([0, 0, 1, 1], "int8"), half, np.arange(n),
                 np.zeros(n, "int8"), (105.0, 68.0))


def test_reveal_mask_contiguous_blocks_inside_gaps_label_free():
    match = _toy_match()
    mask = V3.reveal_mask(match, np.random.default_rng(1))
    assert mask.sum() == 3 * 30  # round(0.12 * 250) per gap
    hidden = np.isnan(match.ball).any(axis=1)
    assert not (mask & ~hidden).any()
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    assert (edges == 1).sum() == 3
    other = replace(match, labels=np.full(len(match.half), 2, "int8"))
    assert np.array_equal(mask, V3.reveal_mask(other, np.random.default_rng(1)))


def test_coverage_view_reveals_clean_ball_only_on_mask():
    match = _toy_match()
    clean = np.nan_to_num(match.ball, nan=20.0)
    view, stats = V3.coverage_view(match, clean, np.random.default_rng(1))
    changed = np.isfinite(view.ball).all(axis=1) & np.isnan(match.ball).any(axis=1)
    assert changed.sum() == stats["revealed_frames"] == 90
    assert np.array_equal(view.labels, match.labels)
    assert stats["revealed_share_of_hidden"] == pytest.approx(0.12)


# ---------------------------------------------------------------- S3 touch features
def _touch_match(n=300, seed=5):
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(n // 2), np.full(n // 2, 2)].astype("int8")
    ball = np.cumsum(rng.normal(0, 0.6, size=(n, 2)), axis=0) + 50.0
    ball[40:60] = np.nan
    players = np.concatenate([rng.uniform(0, 105, size=(n, 6, 1)), rng.uniform(0, 68, size=(n, 6, 1)),
                              np.full((n, 6, 1), 3.0)], axis=2)
    players[100:110, :3] = np.nan  # home absent for a while
    return Match("touch", players, ball, np.array([0, 0, 0, 1, 1, 1], "int8"), half, np.arange(n),
                 np.zeros(n, "int8"), (105.0, 68.0))


def test_touch_features_swap_under_home_away_mirror():
    import exp2_views as V
    import exp3_touch as S

    match = _touch_match()
    direct = S.touch_features(match)
    mirrored = S.touch_features(V.mirror_match(match))
    assert np.allclose(mirrored, S.mirror_features(direct), atol=1e-5)
    assert (direct[:, 3] != 0).any() and (direct[:, 0] != direct[:, 1]).any()


def test_touch_mirror_rules_match_exp1_batch_mirror():
    import exp1_net as N
    import exp1_tokens as T
    import exp3_touch as S

    glob = np.random.default_rng(2).normal(size=(7, len(T.GLOBAL_NAMES) + 4))
    with S.extended_globals():
        names, negate, swap = N.GLOBAL_NAMES, N.GLOBAL_NEGATE, N.GLOBAL_SWAP
    assert N.GLOBAL_NAMES == T.GLOBAL_NAMES and len(names) == len(T.GLOBAL_NAMES) + 4
    mirrored = glob.copy()
    mirrored[:, list(negate)] *= -1
    for left, right in swap:
        mirrored[:, [left, right]] = mirrored[:, [right, left]]
    touch = glob[:, len(T.GLOBAL_NAMES):]
    assert np.allclose(mirrored[:, len(T.GLOBAL_NAMES):], S.mirror_features(touch))


def test_touch_window_never_crosses_half_and_missing_ball_caps_distance():
    import exp3_touch as S

    match = _touch_match()
    features = S.touch_features(match)
    steps = _sample(match.half)
    window_hidden = [k for k, f in enumerate(steps) if np.isnan(match.ball[max(f - 4, 0):f + 1]).all()]
    assert (features[window_hidden, :2] == S.DIST_CAP).all()
    assert window_hidden and (features[window_hidden, 2:] == 0).all()


def _sample(half):
    from exp1_common import sample_indices

    return sample_indices(half)
