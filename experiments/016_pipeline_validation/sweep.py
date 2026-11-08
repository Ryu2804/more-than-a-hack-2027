"""One-at-a-time sweeps of the pipeline's design choices on the three held-out games (harness.py)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import harness as H

GRID = {
    "gate_r": [0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 1e9],
    "kappa": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    "team": ["hidden", "real"],
    "dead_thr": [0.3, 0.4, 0.5, 0.6, 0.7],
    "min_dead_s": [0.0, 1.0, 2.0, 4.0, 8.0],
    "shift": [-0.5, 0.0, 0.5, 1.0],
    "tau": [1.0, 2.0, 3.0, 5.0],
    "rate_scale": [0.7, 0.85, 1.0, 1.15, 1.3],
}
base = H.run({})
b = H.summarise(base); bs = [r["spell_f1"] for r in base]
print(f"BASE  spellF1 {b['spell_f1']:.3f} (per game {[round(x,3) for x in bs]}) macroF1 {b['macro_f1']:.3f} team {b['team_acc']:.3f}")
results = {}
for key, vals in GRID.items():
    print(f"\n== {key}")
    for v in vals:
        rows = H.run({key: v}); s = H.summarise(rows); ps = [r["spell_f1"] for r in rows]
        wins = sum(p > q for p, q in zip(ps, bs))
        tag = "  <== base" if v == H.DEFAULT[key] else ""
        print(f"  {str(v):>7s}: spellF1 {s['spell_f1']:.3f} ({s['spell_f1']-b['spell_f1']:+.3f}, games better {wins}/3) macroF1 {s['macro_f1']:.3f} team {s['team_acc']:.3f} transF1 {s['trans_f1_25']:.3f} spells {[r['pred_spells'] for r in rows]} D {np.mean([r['dead_share'] for r in rows]):.1%}{tag}")
        results.setdefault(key, []).append(dict(value=v, spell_f1=s["spell_f1"], per_game=ps, macro_f1=s["macro_f1"], team_acc=s["team_acc"], trans_f1=s["trans_f1_25"]))
json.dump(dict(base=dict(summary=b, per_game=bs), sweeps=results), open(Path(__file__).resolve().parent / "sweep_results.json", "w"), indent=1)
