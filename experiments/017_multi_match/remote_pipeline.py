"""Sandbox job: build features for many other matches, train grouped-fold models, save held-out predictions (+ optional final ensemble).

Runs on the marimo/molab sandbox. Public data only: Metrica sample games (3) and SkillCorner open data (20 matches).

    python remote_pipeline.py --out OUT [--train metrica,skillcorner] [--folds 6] [--seeds 3] [--steps 1500] [--final] [train_model flags]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import common
import decoder as dc
import features as ft
import other_games as og
import skillcorner_games as sg
import idsse_games as ig
import train_model as tm

FPS = 25


def build_one(args):
    src, gid, outdir = args
    path = Path(outdir) / f"{src}_{gid}.npz"
    if path.exists():
        return str(path)
    d = {"metrica": og.load, "skillcorner": sg.load, "idsse": ig.load}[src](gid)
    P, B, T, half, y = d["P"], d["B"], d["T"], d["half"], d["y"]
    B = ft.fill_ball_gaps(B, int(os.environ.get("FILL_GAP", "0")))
    F = ft.build(P, B, T, half)
    ix = np.flatnonzero(half > 0)[::tm.STRIDE]
    lab = y >= 0
    mins = lab.sum() / FPS / 60
    n_poss = sum(len(tm.runs(y == t)[0]) for t in (0, 1))
    extra = {}
    if ft.USE_PP:
        pp, ppm = ft.player_set(P, B, T, half, ix); extra = dict(pp=pp.astype(np.float16), ppm=ppm)
    np.savez_compressed(path, **extra, ix=ix, player=F["player"][:, ix], ball=F["ball"][:, ix], valid=F["pvalid"][:, ix], y_s=y[ix].astype(np.int8), half_s=half[ix].astype(np.int8),
                        y=y.astype(np.int8), half=half.astype(np.int8), dist=dc.nearest_distances(P, B, T).astype(np.float32),
                        poss_per_min=n_poss / mins, dead_share=float(np.mean(y[lab] == 2)))
    return str(path)


def load_game(path):
    z = np.load(path)
    g = dict(ix=z["ix"], player=z["player"], ball=z["ball"], valid=z["valid"], y=z["y_s"].astype(int), half=z["half_s"].astype(int),
             poss_per_min=float(z["poss_per_min"]), dead_share=float(z["dead_share"]))
    if "pp" in z.files: g["pp"], g["ppm"] = z["pp"].astype(np.float32), z["ppm"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games-dir", default=None)
    ap.add_argument("--train", default="metrica,skillcorner")
    ap.add_argument("--sk", type=int, default=20, help="number of SkillCorner matches")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dil", default=None); ap.add_argument("--crop", type=int, default=None); ap.add_argument("--ch", type=int, default=None)
    ap.add_argument("--mask-full", type=float, default=None); ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--final", action="store_true"); ap.add_argument("--idsse", type=int, default=0, help="number of IDSSE/DFL matches to add")
    a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); (out / "preds").mkdir(parents=True, exist_ok=True)
    gdir = Path(a.games_dir or (out.parent / "games")); gdir.mkdir(parents=True, exist_ok=True)
    tm.SEEDS = tuple(range(a.seeds))
    if a.dil: tm.DIL = tuple(int(x) for x in a.dil.split(","))
    if a.crop: tm.CROP = a.crop
    if a.ch: tm.CH = a.ch
    if a.mask_full is not None: tm.MASK_FULL = a.mask_full
    if a.noise is not None: tm.NOISE = a.noise
    specs = [("metrica", g, str(gdir)) for g in (1, 2, 3)] + [("skillcorner", m, str(gdir)) for m in sg.MATCH_IDS[:a.sk]] + [("idsse", m, str(gdir)) for m in list(ig.MATCHES)[:a.idsse]]
    with Pool(min(8, os.cpu_count() or 4)) as pool:
        paths = pool.map(build_one, specs, chunksize=1)
    names = [Path(p).stem for p in paths]
    print(f"built {len(names)} games | {time.time() - t0:.0f}s", flush=True)
    G = {n: load_game(p) for n, p in zip(names, paths)}
    allowed = set(a.train.split(","))
    core = sorted(n for n in names if not n.startswith('idsse_'))
    fold_of = {n: i % max(a.folds, 1) for i, n in enumerate(core)}
    fold_of.update({n: i % max(a.folds, 1) for i, n in enumerate(sorted(n for n in names if n.startswith('idsse_')))})   # core folds unchanged when IDSSE is added
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    for k in range(a.folds):
        held = [n for n in names if fold_of[n] == k]
        tr = {n: G[n] for n in names if fold_of[n] != k and n.split("_")[0] in allowed}
        mu, sd = tm.standardiser(tr); inputs = tm.make_inputs(mu, sd)
        nets = [tm.train(tr, inputs, s, dev, steps=a.steps) for s in tm.SEEDS]
        rate = float(np.mean([v["poss_per_min"] for v in tr.values()])); dshare = float(np.mean([v["dead_share"] for v in tr.values()]))
        for n in held:
            g = G[n]; res = {}
            for tag, mask in (("hidden", True), ("real", False)):
                pr = [tm.predict(net, inputs, g["player"], g["ball"], g["valid"], mask, dev, g.get("pp"), g.get("ppm")) for net in nets]
                res[f"pdead_{tag}"] = np.mean([p[0] for p in pr], 0).astype(np.float32); res[f"pteam_{tag}"] = np.mean([p[1] for p in pr], 0).astype(np.float32)
                if tm.CHANGE_HEAD: res[f"pchange_{tag}"] = np.mean([p[2] for p in pr], 0).astype(np.float32)
            np.savez(out / "preds" / f"{n}.npz", train_rate=rate, train_dead_share=dshare, fold=k, n_train=len(tr), **res)
        print(f"fold {k}: held {held} | trained on {len(tr)} games | {time.time() - t0:.0f}s", flush=True)
    if a.final:
        tr = {n: G[n] for n in names if n.split("_")[0] in allowed}
        mu, sd = tm.standardiser(tr); inputs = tm.make_inputs(mu, sd); (out / "final").mkdir(exist_ok=True)
        for s in tm.SEEDS:
            net = tm.train(tr, inputs, s, dev, steps=a.steps)
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out / "final" / f"tcn_seed{s}.pt")
        rates = [v["poss_per_min"] for v in tr.values()]; dshares = [v["dead_share"] for v in tr.values()]
        np.savez(out / "final" / "meta.npz", use_pp=bool(ft.USE_PP), mu=mu, sd=sd, dil=np.array(tm.DIL), crop=tm.CROP, rate_mean=float(np.mean(rates)), rate_sd=float(np.std(rates, ddof=1)), dead_share_mean=float(np.mean(dshares)))
        json.dump(dict(games=len(tr), poss_per_min={n: v["poss_per_min"] for n, v in tr.items()}, dead_share={n: v["dead_share"] for n, v in tr.items()}), open(out / "final" / "training_report.json", "w"), indent=1)
        print("final ensemble saved", flush=True)
    print("ALL DONE", f"| {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
