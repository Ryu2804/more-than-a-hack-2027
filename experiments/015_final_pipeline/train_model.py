"""Training code for the supervised dead-ball / possession model. Runs unchanged on a local GPU or in a marimo/molab sandbox.

Uses ONLY other matches (Metrica public sample games 1-3, https://github.com/metrica-sports/sample-data): nothing from the
provided match is read here. Writes weights + the standardisation / calibration constants the pipeline needs.

    python train_model.py --out weights            # validation (leave-one-game-out) + final ensemble
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    sys.path.insert(0, str(p))

import features as ft  # noqa: E402
import other_games as og  # noqa: E402

FPS, STRIDE = 25, 5
GAME_IDS = (1, 2, 3)
SEEDS = (0, 1, 2)
CH, MASK_FULL, NOISE = 64, 0.5, 0.0   # channel width, probability a training crop hides the ball entirely, input-noise std (standardised units)
DIL = (1, 2, 4, 8, 16, 1)          # dilations of the TCN blocks (receptive field = 1 + 4 * sum(DIL) steps at 5 fps)
CROP = 384


def runs(mask):
    e = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


class TCN2(nn.Module):
    def __init__(self, cin, ch=None, dil=None, k=5, p=0.15):
        ch = ch or CH
        super().__init__()
        dil = tuple(dil) if dil is not None else DIL
        self.inp = nn.Conv1d(cin, ch, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(ch, ch, k, padding=d * (k // 2), dilation=d), nn.GELU(), nn.Dropout(p)) for d in dil])
        self.norm = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dil])
        self.head = nn.Conv1d(ch, 2, 1)             # channel 0: dead ball, channel 1: team a in possession

    def forward(self, x):
        h = self.inp(x)
        for blk, nm in zip(self.blocks, self.norm):
            h = h + blk(nm(h))
        return self.head(h)


NCH = len(ft.PLAYER_FEATS) + 1 + len(ft.BALL_FEATS)


def build_games(ids=GAME_IDS):
    """Per game: stride-5 features and labels, plus the full-rate possession rate and stoppage stats of its derived labels."""
    games = {}
    for n in ids:
        g = og.load(n)
        F = ft.build(g["P"], g["B"], g["T"], g["half"])
        ix = np.flatnonzero(g["half"] > 0)[::STRIDE]
        y_full, lab = g["y"], g["y"] >= 0
        mins = lab.sum() / FPS / 60
        n_poss = sum(len(runs(y_full == t)[0]) for t in (0, 1))
        s, e = runs(y_full == 2)
        games[n] = dict(ix=ix, player=F["player"][:, ix], ball=F["ball"][:, ix], valid=F["pvalid"][:, ix], y=y_full[ix].astype(int), half=g["half"][ix],
                        poss_per_min=n_poss / mins, dead_share=float(np.mean(y_full[lab] == 2)), stoppage_median_s=float(np.median((e - s) / FPS)))
    return games


def standardiser(games):
    pl = np.concatenate([G["player"][0] for G in games.values()])
    return np.nanmean(pl, 0), np.nanstd(pl, 0) + 1e-6


def make_inputs(mu, sd):
    def inputs(pl, bl, valid, mask_ball):
        f = np.nan_to_num((pl - mu) / sd).astype(np.float32)
        b = np.nan_to_num(np.where(mask_ball[:, None], 0.0, bl)).astype(np.float32)
        return np.concatenate([f, valid[:, None].astype(np.float32), b], 1)
    return inputs


def random_mask(n, rng):
    if rng.random() < MASK_FULL:
        return np.ones(n, bool)
    m = np.zeros(n, bool); i = 0
    while i < n:
        ln = int(rng.integers(8, 160))
        if rng.random() < 0.5:
            m[i:i + ln] = True
        i += ln
    return m


def train(games, inputs, seed, dev, steps=1500, bs=32, crop=None):
    crop = crop or CROP
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    seqs = [(G, np.flatnonzero(G["half"] == h)) for G in games.values() for h in (1, 2)]
    seqs = [(G, s) for G, s in seqs if len(s) > crop + 16]
    net = TCN2(NCH, ch=CH, dil=DIL).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    bce = nn.functional.binary_cross_entropy_with_logits
    for _ in range(steps):
        X, Yd, Yt, Md, Mt = [], [], [], [], []
        for _ in range(bs):
            G, s = seqs[int(rng.integers(len(seqs)))]
            st = int(rng.integers(0, len(s) - crop)); sl = s[st:st + crop]; v = int(rng.integers(0, 2))
            y = G["y"][sl]
            X.append(inputs(G["player"][v][sl], G["ball"][v][sl], G["valid"][v][sl], random_mask(crop, rng)))
            Yd.append((y == 2).astype(np.float32)); Yt.append((y == v).astype(np.float32))
            Md.append((y >= 0).astype(np.float32)); Mt.append(((y == 0) | (y == 1)).astype(np.float32))
        X = np.stack(X)
        if NOISE > 0:
            X[:, :, :len(ft.PLAYER_FEATS)] += NOISE * rng.standard_normal(X[:, :, :len(ft.PLAYER_FEATS)].shape).astype(np.float32)
        X = torch.tensor(X).permute(0, 2, 1).to(dev)
        Yd, Yt, Md, Mt = [torch.tensor(np.stack(a)).to(dev) for a in (Yd, Yt, Md, Mt)]
        out = net(X)
        loss = (bce(out[:, 0], Yd, reduction="none") * Md).sum() / Md.sum().clamp(min=1) + (bce(out[:, 1], Yt, reduction="none") * Mt).sum() / Mt.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, inputs, pl, bl, valid, mask_all, dev):
    outs = []
    for v in (0, 1):
        x = torch.tensor(inputs(pl[v], bl[v], valid[v], np.full(len(pl[v]), mask_all))).T[None].to(dev)
        outs.append(torch.sigmoid(net(x))[0].cpu().numpy())
    return 0.5 * (outs[0][0] + outs[1][0]), 0.5 * (outs[0][1] + 1 - outs[1][1])     # p_dead, p_team_H


def logo_eval(games, dev, steps):
    """Leave-one-game-out: TCN vs per-step gradient boosting, ball hidden and available."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    rows = []
    for te in games:
        tr = {k: v for k, v in games.items() if k != te}
        mu, sd = standardiser(tr); inputs = make_inputs(mu, sd)
        G = games[te]; y = G["y"]; lab = y >= 0; alive = (y == 0) | (y == 1)
        nets = [train(tr, inputs, s, dev, steps=steps) for s in SEEDS]
        for mask_ball, cond in ((True, "ball hidden"), (False, "ball available")):
            pr = [predict(n, inputs, G["player"], G["ball"], G["valid"], mask_ball, dev) for n in nets]
            tcn = (np.mean([p[0] for p in pr], 0), np.mean([p[1] for p in pr], 0))
            X, yd, yt, mt = [], [], [], []
            for H in tr.values():
                for v in (0, 1):
                    m = H["y"] >= 0
                    X.append(inputs(H["player"][v], H["ball"][v], H["valid"][v], np.full(len(H["y"]), mask_ball))[m])
                    yv = H["y"][m]; yd.append(yv == 2); yt.append(yv == v); mt.append((yv == 0) | (yv == 1))
            X, yd, yt, mt = map(np.concatenate, (X, yd, yt, mt))
            gd = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0).fit(X, yd)
            gt = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0).fit(X[mt], yt[mt])
            pd_, pt = [], []
            for v in (0, 1):
                f = inputs(G["player"][v], G["ball"][v], G["valid"][v], np.full(len(y), mask_ball))
                pd_.append(gd.predict_proba(f)[:, 1]); pt.append(gt.predict_proba(f)[:, 1])
            gbm = (0.5 * (pd_[0] + pd_[1]), 0.5 * (pt[0] + 1 - pt[1]))
            for name, (pdn, ptn) in (("TCN", tcn), ("gradient boosting", gbm)):
                rows.append(dict(train="+".join(map(str, tr)), test=f"game {te}", condition=cond, model=name,
                                 auc_dead=float(roc_auc_score(y[lab] == 2, pdn[lab])), acc_dead=float(accuracy_score(y[lab] == 2, pdn[lab] > 0.5)),
                                 acc_team_alive=float(accuracy_score(y[alive] == 0, ptn[alive] > 0.5))))
    return rows


def fold_predictions(games, dev, steps, out):
    """Held-out predictions per fold: models trained on the OTHER games only (3 seeds), predicted on the held-out game at stride 5
    with the ball hidden and with the ball real. Saved as npz so the full pipeline can be validated offline on public data."""
    for te in games:
        tr = {k: v for k, v in games.items() if k != te}
        mu, sd = standardiser(tr); inputs = make_inputs(mu, sd)
        nets = [train(tr, inputs, s, dev, steps=steps) for s in SEEDS]
        G = games[te]; res = {}
        for tag, mask in (("hidden", True), ("real", False)):
            pr = [predict(n, inputs, G["player"], G["ball"], G["valid"], mask, dev) for n in nets]
            res[f"pdead_{tag}"] = np.mean([p[0] for p in pr], 0).astype(np.float32)
            res[f"pteam_{tag}"] = np.mean([p[1] for p in pr], 0).astype(np.float32)
        rate = float(np.mean([v["poss_per_min"] for v in tr.values()])); dshare = float(np.mean([v["dead_share"] for v in tr.values()]))
        np.savez(Path(out) / f"fold_game{te}.npz", ix=G["ix"], half=G["half"], train_rate=rate, train_dead_share=dshare, **res)
        print(f"fold game {te} saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "weights"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--no-logo", action="store_true")
    ap.add_argument("--folds", action="store_true", help="only write held-out per-fold predictions (fold_game*.npz) and exit")
    ap.add_argument("--dil", default=None, help="comma-separated dilations, e.g. 1,2,4,8,16,32,64,1")
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ch", type=int, default=None)
    ap.add_argument("--mask-full", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    global DIL, CROP, SEEDS, CH, MASK_FULL, NOISE
    SEEDS = tuple(range(a.seeds))
    if a.ch: CH = a.ch
    if a.mask_full is not None: MASK_FULL = a.mask_full
    if a.noise is not None: NOISE = a.noise
    if a.dil: DIL = tuple(int(x) for x in a.dil.split(","))
    if a.crop: CROP = a.crop
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""), flush=True)
    games = build_games()
    print({n: dict(poss_per_min=round(G["poss_per_min"], 2), dead_share=round(G["dead_share"], 3), stoppage_median_s=round(G["stoppage_median_s"], 1)) for n, G in games.items()}, f"| {time.time() - t0:.0f}s", flush=True)
    if a.folds:
        fold_predictions(games, dev, a.steps, out); print("folds done", f"| {time.time() - t0:.0f}s total", flush=True); return
    result = dict(device=dev, steps=a.steps, seeds=len(SEEDS), ch=CH, mask_full=MASK_FULL, noise=NOISE, dil=list(DIL), crop=CROP, receptive_field_steps=1 + 4 * sum(DIL), games={str(n): {k: v for k, v in G.items() if k in ("poss_per_min", "dead_share", "stoppage_median_s")} for n, G in games.items()})
    if not a.no_logo:
        result["logo"] = logo_eval(games, dev, a.steps)
        print("leave-one-game-out done", f"| {time.time() - t0:.0f}s", flush=True)
    mu, sd = standardiser(games); inputs = make_inputs(mu, sd)
    for s in SEEDS:
        net = train(games, inputs, s, dev, steps=a.steps)
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out / f"tcn_seed{s}.pt")
    rates = [G["poss_per_min"] for G in games.values()]
    np.savez(out / "meta.npz", mu=mu, sd=sd, dil=np.array(DIL), crop=CROP, rate_mean=float(np.mean(rates)), rate_sd=float(np.std(rates, ddof=1)))
    (out / "training_report.json").write_text(json.dumps(result, indent=1))
    print("final ensemble saved", f"| {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
