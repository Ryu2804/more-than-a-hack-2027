"""BAR-ATH label-free detector inputs: ATD ball and players, per-frame homography, run splits,
auto-labels (PREREG 3.1-3.2) and the sequential video reader.

Hygiene: every array stays inside the calling process. Summaries are aggregate counts and shares;
nothing per-frame is printed. No frame, crop or image is ever written to disk.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import sys
import time
from typing import Callable, Iterator

import numpy as np

import exp6_common as C
from exp1_common import JD

N_FRAMES = 199_319
IMAGE_W, IMAGE_H = 1920, 1080
FPS = 25
PITCH = (105.3, 68.0)  # metres, as JD.load_target scales the normalised ATD
SPLIT_MOD = 5
TRAIN_RESIDUES = (0, 1, 2)
VAL_RESIDUE = 3
TEST_RESIDUE = 4
LABEL_MARGIN_PX = 8.0
UNGRADED, TRAIN, VAL, TEST, HIDDEN = -1, 0, 1, 2, 3
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test", HIDDEN: "hidden"}


# ---------------------------------------------------------------- competition helpers (by import)
def _target_repo_module(name: str):
    """Import ``games`` or ``starter/load_homography`` from the clone that holds BAR-ATH."""
    repo = C.C1.TARGET_REPO
    for folder in (repo, repo / "starter"):
        if str(folder) not in sys.path:
            sys.path.append(str(folder))
    return __import__(name)


def graded_half() -> np.ndarray:
    """Published graded halves (1, 2) per video frame, 0 elsewhere (games.half_video_blocks)."""
    games = _target_repo_module("games")
    half = np.zeros(N_FRAMES, dtype="int8")
    for period, first, last in games.half_video_blocks("BAR-ATH"):
        half[first:last + 1] = period
    return half


def load_homography() -> np.ndarray:
    """(N, 3, 3) normalised pitch -> pixel matrices via the starter loader; one row per frame."""
    loader = _target_repo_module("load_homography")
    data_dir = str(C.C1.TARGET_REPO / "data")
    if not loader.check_alignment(data_dir=data_dir, source="tactical")["agrees"]:
        raise RuntimeError("Homography rows are not one per video frame")
    matrices, _, valid = loader.load(data_dir=data_dir, source="tactical")
    if len(matrices) != N_FRAMES or not bool(valid.all()):
        raise RuntimeError("Tactical homography must be valid on every frame")
    return matrices


# ---------------------------------------------------------------- geometry
def project(matrices: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The starter's ``project`` formula with one 3x3 matrix per leading row (broadcast over points)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    h = np.asarray(matrices, dtype=np.float64)
    h = h.reshape(h.shape[:1] + (1,) * (x.ndim - 1) + (3, 3))
    denominator = h[..., 2, 0] * x + h[..., 2, 1] * y + h[..., 2, 2]
    return ((h[..., 0, 0] * x + h[..., 0, 1] * y + h[..., 0, 2]) / denominator,
            (h[..., 1, 0] * x + h[..., 1, 1] * y + h[..., 1, 2]) / denominator)


def metres_to_pixels(matrices: np.ndarray, xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy_m, dtype=np.float64)
    return project(matrices, xy[..., 0] / PITCH[0], xy[..., 1] / PITCH[1])


def pixels_to_metres(inverse: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pixels -> metres on 105.3 x 68.0 through the per-frame inverse homography."""
    x, y = project(inverse, u, v)
    return np.stack([x * PITCH[0], y * PITCH[1]], axis=-1)


def inside_image(u: np.ndarray, v: np.ndarray, margin: float = 0.0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return ((u >= margin) & (u <= IMAGE_W - 1 - margin) & (v >= margin) & (v <= IMAGE_H - 1 - margin)
                & np.isfinite(u) & np.isfinite(v))


def auto_labels(ball_m: np.ndarray, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected ATD ball (pixels) and whether it lies inside the image with the 8 px margin."""
    u, v = metres_to_pixels(matrices, ball_m)
    keep = inside_image(u, v, LABEL_MARGIN_PX)
    uv = np.where(keep[:, None], np.stack([u, v], axis=1), np.nan)
    return uv, keep


def pitch_bbox(matrices: np.ndarray) -> np.ndarray:
    """(N, 4) int [x_min, y_min, x_max, y_max) of the projected pitch rectangle, clipped to the image."""
    corners_x = np.array([0.0, 1.0, 1.0, 0.0])[None].repeat(len(matrices), 0)
    corners_y = np.array([0.0, 0.0, 1.0, 1.0])[None].repeat(len(matrices), 0)
    u, v = project(matrices, corners_x, corners_y)
    box = np.stack([np.floor(u.min(1)), np.floor(v.min(1)), np.ceil(u.max(1)) + 1, np.ceil(v.max(1)) + 1], 1)
    box = np.nan_to_num(box, nan=0.0)
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, IMAGE_W)
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, IMAGE_H)
    return box.astype("int32")


# ---------------------------------------------------------------- splits (label-free)
def split_frames(visible: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Per-frame split: visible runs numbered k in time order within each half; k mod 5 decides."""
    split = np.full(len(half), UNGRADED, dtype="int8")
    split[(half > 0) & ~visible] = HIDDEN
    for period in np.unique(half[half > 0]):
        index = np.flatnonzero(half == period)
        for k, (start, stop) in enumerate(JD.runs(visible[index])):
            residue = k % SPLIT_MOD
            code = TRAIN if residue in TRAIN_RESIDUES else VAL if residue == VAL_RESIDUE else TEST
            split[index[start:stop]] = code
    return split


def split_summary(split: np.ndarray, half: np.ndarray, keep: np.ndarray) -> dict:
    """Aggregate frame and run counts per split (label-free)."""
    out = {}
    for code, name in SPLIT_NAMES.items():
        mask = split == code
        runs = sum(len(JD.runs(mask[half == p])) for p in np.unique(half[half > 0]))
        out[name] = {"frames": int(mask.sum()), "runs": int(runs), "labelled_frames": int((mask & keep).sum())}
    return out


# ---------------------------------------------------------------- BAR-ATH inputs
@dataclass(frozen=True)
class TargetInputs:
    ball: np.ndarray  # (N, 2) metres, NaN where ATD has no ball
    players: np.ndarray  # (N, K, 2) metres, team slots only (others NaN)
    half: np.ndarray  # published graded halves (1, 2), 0 elsewhere
    matrices: np.ndarray  # (N, 3, 3) normalised pitch -> pixels
    split: np.ndarray  # per-frame split code
    label_uv: np.ndarray  # (N, 2) projected ATD ball, NaN unless kept
    keep: np.ndarray  # label kept (inside the image with the margin)
    bbox: np.ndarray  # (N, 4) pitch image bounding box

    @property
    def visible(self) -> np.ndarray:
        return np.isfinite(self.ball).all(axis=1)


def load_inputs() -> TargetInputs:
    """Tactical ATD through exp_1's loader (fusion off), homography through the starter loader."""
    import exp1_target as X1

    match = X1.load_target_match()
    half = graded_half()
    if not np.array_equal(match.half > 0, half > 0) or len(match.half) != N_FRAMES:
        raise RuntimeError("Graded frames differ between the ATD loader and games.py")
    matrices = load_homography()
    ball = match.ball.astype("float64")
    visible = np.isfinite(ball).all(axis=1)
    label_uv, keep = auto_labels(np.where(visible[:, None], ball, np.nan), matrices)
    keep &= half > 0
    return TargetInputs(ball=ball, players=match.players[:, :, :2].astype("float64"), half=half,
                        matrices=matrices, split=split_frames(visible, half),
                        label_uv=np.where(keep[:, None], label_uv, np.nan), keep=keep,
                        bbox=pitch_bbox(matrices))


def g1(inputs: TargetInputs) -> dict:
    """Gate 1: share of ATD-visible graded balls projecting inside the image (8 px margin)."""
    visible = inputs.visible & (inputs.half > 0)
    return {"visible_graded": int(visible.sum()), "inside_with_margin": int((inputs.keep & visible).sum()),
            "share": float((inputs.keep & visible).sum() / max(1, visible.sum()))}


def input_summary(inputs: TargetInputs) -> dict:
    """Aggregates only: split counts, label shares, bounding-box size and player projection share."""
    graded = inputs.half > 0
    width = (inputs.bbox[:, 2] - inputs.bbox[:, 0])[graded]
    height = (inputs.bbox[:, 3] - inputs.bbox[:, 1])[graded]
    sample = np.flatnonzero(graded)[::250]
    u, v = metres_to_pixels(inputs.matrices[sample], inputs.players[sample])
    present = np.isfinite(inputs.players[sample]).all(axis=-1)
    return {"graded_frames": int(graded.sum()), "hidden_graded": int((inputs.split == HIDDEN).sum()),
            "splits": split_summary(inputs.split, inputs.half, inputs.keep), "g1": g1(inputs),
            "bbox_width_px": {"median": float(np.median(width)), "min": int(width.min()), "max": int(width.max())},
            "bbox_height_px": {"median": float(np.median(height)), "min": int(height.min()),
                               "max": int(height.max())},
            "player_projection_inside_share": float(inside_image(u, v)[present].mean())}


# ---------------------------------------------------------------- sequential video
def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[:-shift] |= mask[shift:]
        out[shift:] |= mask[:-shift]
    return out


class VideoReader:
    """Sequential cv2 reader: ``grab()`` every frame, ``retrieve()`` only where needed (no seeking)."""

    def __init__(self, path=C.VIDEO, n_frames: int = N_FRAMES) -> None:
        import cv2

        self.cv2, self.path, self.n_frames = cv2, path, n_frames

    def metadata(self) -> dict:
        capture = self.cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise RuntimeError("Tactical video could not be opened")
            return {"width": int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": float(capture.get(self.cv2.CAP_PROP_FPS)),
                    "frame_count": int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))}
        finally:
            capture.release()

    def frames(self, needed: np.ndarray, check_count: bool = False) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, BGR frame) for needed frames, reading every frame in order."""
        if len(needed) != self.n_frames:
            raise ValueError("Needed mask must cover every video frame")
        meta = self.metadata()
        if (meta["width"], meta["height"]) != (IMAGE_W, IMAGE_H) or meta["frame_count"] != self.n_frames:
            raise RuntimeError("Tactical video metadata differs from DATA.md")
        last = self.n_frames - 1 if check_count else int(np.flatnonzero(needed).max(initial=-1))
        capture = self.cv2.VideoCapture(str(self.path))
        try:
            for index in range(last + 1):
                if not capture.grab():
                    raise RuntimeError("Tactical video ended early")
                if needed[index]:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise RuntimeError("Frame retrieve failed")
                    yield index, frame
            if check_count and capture.grab():
                raise RuntimeError("Tactical video has more frames than expected")
        finally:
            capture.release()


class ArraySource:
    """In-memory frame source with the VideoReader interface (synthetic tests only)."""

    def __init__(self, video: np.ndarray) -> None:
        self.video, self.n_frames = video, len(video)

    def frames(self, needed: np.ndarray, check_count: bool = False) -> Iterator[tuple[int, np.ndarray]]:
        for index in np.flatnonzero(needed):
            yield int(index), self.video[index]


def triplets(frames: Iterator[tuple[int, np.ndarray]], targets: np.ndarray,
             transform: Callable = lambda frame: frame) -> Iterator[tuple[int, tuple]]:
    """Yield (t, (frame t-1, frame t, frame t+1)) for every target t, from a 3-frame ring buffer."""
    ring: deque = deque(maxlen=3)
    for index, frame in frames:
        ring.append((index, transform(frame)))
        if len(ring) == 3 and ring[0][0] == index - 2 and ring[1][0] == index - 1 and targets[index - 1]:
            yield index - 1, (ring[0][1], ring[1][1], ring[2][1])


def decode_check(reader: VideoReader, count: int = 300, probes: tuple[int, ...] = (0, 150, 299)) -> dict:
    """Timing of ``count`` sequential decodes + whether two reads give byte-identical probe frames."""
    needed = np.zeros(reader.n_frames, dtype=bool)
    needed[:count] = True
    digests, seconds = [], []
    for _ in range(2):
        start, found = time.time(), {}
        for index, frame in reader.frames(needed):
            if index in probes:
                found[index] = hashlib.sha256(frame.tobytes()).hexdigest()
        seconds.append(time.time() - start)
        digests.append([found[p] for p in probes])
    return {"frames_read": count, "seconds_per_read": [round(s, 2) for s in seconds],
            "frames_per_second": round(count / min(seconds), 1), "deterministic": digests[0] == digests[1]}
