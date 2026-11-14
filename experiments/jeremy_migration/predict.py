#!/usr/bin/env python3
"""Migrate jeremy SetSeq pipeline to per-match IDSSE predictions."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORKTREE = "/tmp/opencode/jeremy-worktree/experiments/exp_1"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "judge"))
os.environ.setdefault("MTAH_REPO", "/home/rasyaarya/lomba/more_than_a_hack/more-than-a-hack-2027")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(WORKTREE, "cache", "numba"))

import exp1_common as C  # noqa: E402
import exp1_protocol as P  # noqa: E402
import exp1_decode as D  # noqa: E402
import exp1_tokens as T  # noqa: E402

C.verify_judge()

OUT = Path(os.environ.get("JEREMY_OUT", "/tmp/opencode/jeremy_migration_out"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pred").mkdir(exist_ok=True)
(OUT / "gt").mkdir(exist_ok=True)

# Build token cache if needed
missing = any(not (C.TOKEN_DIR / f"{name}_r0.npz").exists() for name in C.MATCH_IDS)
if missing:
    print("building token cache...", flush=True)
    P.build_token_cache()

with open(os.path.join(WORKTREE, "results", "variant_setseq_bd.json")) as f:
    final_config = D.DecoderConfig(**json.load(f)["final_config"])
print("final config", final_config, flush=True)

LABELS = {0: "H", 1: "A", 2: "D", 3: "X"}


def write_csv(path, codes):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "label"])
        for i, c in enumerate(codes):
            w.writerow([i, LABELS[int(c)]])


for target in C.MATCH_IDS:
    out_pred = OUT / "pred" / f"{target}_pred.csv"
    if out_pred.exists():
        print(target, "already done, skipping", flush=True)
        continue
    t0 = time.time()
    train_names = [n for n in C.MATCH_IDS if n != target]
    tok = P.load_tokens(target, 0)
    item = {
        "tokens": tok["tokens"],
        "mask": tok["mask"],
        "glob": tok["glob"],
        "half": tok["half"],
    }
    preds = P.fit_predict("setseq_noaug", train_names, {target: item})
    probability, boundary = preds[target]
    codes_5 = D.decode(probability, boundary, tok["half"], final_config)
    match = C.load_degraded(target)
    full = C.expand(codes_5, tok["indices"], match.half).astype(np.int8)
    base = 10000
    n_frames = int(match.frame.max()) - base + 1
    aligned = np.full(n_frames, 3, dtype=np.int8)
    for idx, frame_id in enumerate(match.frame):
        if base <= frame_id < base + n_frames:
            aligned[int(frame_id) - base] = full[idx]
    write_csv(out_pred, aligned)
    print(target, "done in", round(time.time() - t0, 1), "spell_proxy_n/a", flush=True)

print("all jeremy predictions written to", OUT / "pred")