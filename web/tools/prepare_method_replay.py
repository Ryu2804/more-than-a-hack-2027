"""Export cached method observations locally; never write licensed data into the repo."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ARTIFACT_SHA256 = "cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8"
FIRST_FRAME = 11185
WINDOW_FRAMES = 7500
TOTAL_FRAMES = 199319
STEP = 5
TARGET_KEYS = ("players", "ball", "teams", "sampled", "blocks", "blocks25", "n_frames", "X", "fps")
LABELS = ("H", "A", "D", "X")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outside_output(path, repo, force):
    if not path.is_absolute():
        raise ValueError("--output must be an absolute path outside the repository")
    resolved = path.resolve()
    for root in (repo.resolve(), Path(__file__).resolve().parents[2]):
        if path.absolute().is_relative_to(root) or resolved.is_relative_to(root):
            raise ValueError("Refusing to export licensed observations inside the repository")
    if path.is_symlink():
        raise ValueError("--output must not be a symbolic link")
    if resolved.exists() and not force:
        raise FileExistsError("Output exists; use --force to replace it")
    if not resolved.parent.is_dir():
        raise ValueError("The output parent directory must already exist")
    if resolved.exists() and not resolved.is_file():
        raise ValueError("--output must name a regular file")
    return resolved


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def production_modules(wave5):
    # The research scripts use unqualified imports shared with earlier waves.
    names = ("support", "campaign", "method_replay_fusion", "method_replay_decoder",
             "wave4_official_metrics", "wave5_base_decoder")
    saved = {name: sys.modules.get(name) for name in names}
    original_path = sys.path[:]
    original_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(wave5), str(wave5.parent / "wave4")]
        sys.modules.pop("campaign", None)
        support = import_file("support", wave5 / "support.py")
        fusion = import_file("method_replay_fusion", wave5 / "adaptive_fusion.py")
        decoder = import_file("method_replay_decoder", wave5 / "segment_decoder.py")
        yield support, fusion, decoder
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def sampled_window(data):
    sampled = data["sampled"]
    if np.any(np.diff(sampled) <= 0):
        raise ValueError("sampled source-frame indices must be strictly increasing")
    n = len(sampled)
    if sampled.shape != (n,) or sampled.dtype.kind not in "iu":
        raise ValueError("sampled must contain integer source-frame indices")
    blocks = np.asarray(data["blocks"])
    blocks25 = np.asarray(data["blocks25"])
    if blocks.shape != (2, 2) or blocks25.shape != blocks.shape:
        raise ValueError("Expected the frozen target's two tracking blocks")
    if blocks.dtype.kind not in "iu" or blocks25.dtype.kind not in "iu":
        raise ValueError("Tracking block bounds must be integers")
    covered = np.zeros(n, dtype=bool)
    previous_end = 0
    for (start, stop), (begin, end) in zip(blocks, blocks25):
        if not (0 <= start < stop <= n and previous_end <= begin < end <= TOTAL_FRAMES):
            raise ValueError("Invalid or overlapping tracking blocks")
        expected = np.arange(begin, end, STEP)
        if not np.array_equal(sampled[start:stop], expected) or np.any(covered[start:stop]):
            raise ValueError("sampled and tracking block expansion disagree")
        covered[start:stop] = True
        previous_end = int(end)
    if not np.all(covered):
        raise ValueError("Tracking samples fall outside their blocks")
    absolute = np.arange(FIRST_FRAME, FIRST_FRAME + WINDOW_FRAMES, STEP)
    indices = np.searchsorted(sampled, absolute)
    if np.any(indices >= n) or not np.array_equal(sampled[indices], absolute):
        raise ValueError("The exact replay window is unavailable in cached tracking")
    return indices


def probability(archive, key, n):
    values = archive[key]
    if values.shape != (n, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain finite (N, 3) probabilities")
    if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError(f"{key} rows must be probability distributions")
    return values


def verify_csv(path, full):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["frame", "label"]:
            raise ValueError("Frozen CSV must have frame,label columns")
        count = 0
        for count, row in enumerate(reader, 1):
            index = count - 1
            if index >= TOTAL_FRAMES or row["frame"] != str(index) or row["label"] != LABELS[int(full[index])]:
                raise ValueError(f"Decoded prediction differs from frozen CSV at frame {index}")
        if count != TOTAL_FRAMES:
            raise ValueError("Frozen CSV does not contain all 199319 frames")


def coordinate(value):
    return value.tolist() if np.all(np.isfinite(value)) else None


def build_replay(repo):
    wave5 = repo / "experiments/exp_farras/wave5"
    wave4 = wave5.parent / "wave4"
    selection_path = wave5 / "out/selection.json"
    report_path = wave5 / "out/holdout_report.json"
    registry_path = wave5 / "out/preregistered.json"
    selection, report, registry = map(read_json, (selection_path, report_path, registry_path))
    hashes = {}

    def record(path, expected=None):
        relative = path.relative_to(repo).as_posix()
        if not path.resolve().is_relative_to(repo):
            raise ValueError(f"Source escapes repository: {relative}")
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input changed: {relative}")
        hashes[relative] = actual
        return actual

    record(selection_path, report["selection_sha256"])
    record(report_path)
    record(registry_path, selection["preregistered_sha256"])
    selected = selection["selected"]
    submitted = report["submissions"]["selected"]
    if selected["family"] != "short_gru25" or selected["decoder"]["name"] != "dead45":
        raise ValueError("Frozen selection is not the requested method")
    if any(submitted[key] != selected[key] for key in ("family", "decoder")):
        raise ValueError("Selection and holdout submission settings differ")
    if (submitted["sha256"] != ARTIFACT_SHA256 or report["reproduced_sha256"] != ARTIFACT_SHA256
            or report["identical_target_rows"] != TOTAL_FRAMES):
        raise ValueError("Holdout report does not identify the exact frozen artifact")
    artifact = wave5 / submitted["path"]
    record(artifact, ARTIFACT_SHA256)
    for family, expected in selection["checkpoint_sha256"].items():
        record(wave5 / "out/models" / (family + ".pt"), expected)

    def frozen(path):
        relative = path.relative_to(repo).as_posix()
        if relative not in registry["sha256"]:
            raise ValueError(f"No preregistered input hash: {relative}")
        record(path, registry["sha256"][relative])

    for path in (wave5 / "support.py", wave5 / "adaptive_fusion.py", wave5 / "segment_decoder.py",
                 wave5 / "run.py", wave4 / "campaign.py", wave4 / "decoder.py",
                 repo / "evaluation/possession_metrics.py"):
        frozen(path)
    data_path = wave5.parent / "wave3/cache/data/target_clean.npz"
    prior_path = wave4 / "cache/prior/target_clean.npz"
    primary_path = wave5 / "cache/predictions/short_tcn/holdout/target_clean.npz"
    frozen(data_path)
    frozen(prior_path)
    # Holdout prediction caches were produced after preregistration. Full CSV equality
    # below authenticates their decoded result; their current bytes are recorded too.
    record(primary_path)
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in TARGET_KEYS}
    n = len(data["sampled"])
    if (n != 31772 or int(data["n_frames"]) != TOTAL_FRAMES or float(data["fps"]) != 5
            or data["players"].shape != (n, 46, 2) or data["ball"].shape != (n, 2)
            or data["X"].shape != (n, 16) or data["teams"].shape != (46,)
            or not np.all(np.isin(data["teams"], (-1, 0, 1)))):
        raise ValueError("Cached target dimensions or team codes differ from the frozen input")
    indices = sampled_window(data)
    with np.load(primary_path, allow_pickle=False) as archive:
        primary = {"probability": probability(archive, "probability", n)}
        for key in ("boundary", "has_boundary"):
            if key in archive.files:
                primary[key] = archive[key]
        if "has_boundary" in primary:
            flag = primary["has_boundary"]
            if flag.shape != () or flag.dtype.kind != "b":
                raise ValueError("Invalid cached boundary provenance")
            primary["has_boundary"] = bool(flag)
    with np.load(prior_path, allow_pickle=False) as archive:
        secondary = {"probability": probability(archive, "hybrid_probability", n), "has_boundary": False}
    with production_modules(wave5) as (support, fusion, decoder):
        for name in support.DEV:
            if name == "target" or name in support.HOLD:
                raise ValueError("Training-statistics split includes a target or holdout")
            frozen(support.base.pack_path(name, "r0" if name.startswith("J03") else "clean"))
        stats = support.training_stats(support.DEV)
        observations = support.base.decoder_observations(data)
        mixed = fusion.combine(selected["family"], {"short_tcn": primary}, None, secondary, observations)
        decoded = decoder.decode(mixed, observations, selected["decoder"], stats)
        full = support.expand(decoded["codes"], data)
        if full.shape != (TOTAL_FRAMES,) or not np.all(np.isin(full, (0, 1, 2, 3))):
            raise ValueError("Production expansion returned invalid labels")
        verify_csv(artifact, full)
        accepted = decoder.dead_mask(mixed["p_dead"], data["blocks"],
                                     threshold=selected["decoder"]["dead_threshold"],
                                     min_s=selected["decoder"]["min_dead_s"], shift=decoded["dead_shift"])
    distances = observations["distances"]
    used = (np.isfinite(data["X"][:, 3]) & (data["X"][:, 3] >= 0.5)
            & np.isfinite(distances).all(axis=1) & (distances >= 0).all(axis=1)
            & (distances.min(axis=1) <= selected["decoder"]["gate_r"]))
    samples = []
    for relative, index in zip(range(0, WINDOW_FRAMES, STEP), indices):
        samples.append({
            "frame": relative,
            "players": [coordinate(value) for value in data["players"][index]],
            "ball": coordinate(data["ball"][index]),
            "primary": primary["probability"][index].tolist(),
            "secondary": secondary["probability"][index].tolist(),
            "mixed": mixed["probability"][index].tolist(),
            "distances": [float(value) if np.isfinite(value) else None for value in distances[index]],
            "proximityUsed": bool(used[index]),
            "deadCandidate": bool(mixed["probability"][index, 2] > 0.45),
            "deadAccepted": bool(accepted[index]),
            "final": LABELS[int(full[FIRST_FRAME + relative])],
        })
    exporter = Path(__file__).resolve()
    if exporter.is_relative_to(repo):
        record(exporter)
    return {"schemaVersion": 1, "match": "BAR-ATH", "methodId": "farras_wave5_short_gru25_dead45",
            "firstFrame": FIRST_FRAME, "fps": 25, "frames": WINDOW_FRAMES, "sampleStep": STEP,
            "pitch": {"length": 105.3, "width": 68}, "teams": data["teams"].astype(int).tolist(),
            "samples": samples,
            "provenance": {"artifactSha256": ARTIFACT_SHA256, "sourceHashes": hashes,
                           "fullPredictionVerified": True, "containsGroundTruth": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Absolute JSON path outside the repository")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--force", action="store_true", help="Replace an existing local export")
    args = parser.parse_args()
    temporary = None
    try:
        repo = args.repo.resolve(strict=True)
        output = outside_output(args.output, repo, args.force)
        replay = build_replay(repo)
        payload = json.dumps(replay, allow_nan=False, separators=(",", ":")) + "\n"
        output = outside_output(args.output, repo, args.force)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                                         prefix=".method-replay-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        if args.force:
            os.replace(temporary, output)
        else:
            # A hard link publishes atomically without clobbering a concurrent writer.
            os.link(temporary, output)
            temporary.unlink()
        temporary = None
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Replay export failed: {error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("Exported 1500 cached samples; all 199319 frozen prediction labels verified. Keep this JSON local.")


if __name__ == "__main__":
    main()
