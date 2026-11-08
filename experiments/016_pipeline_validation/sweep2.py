"""Round 2: refine gate radius, then re-sweep the other knobs around the adopted combination (gate + team=real)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import harness as H

def show(tag, cfg, ref=None):
    rows = H.run(cfg); s = H.summarise(rows); ps = [r["spell_f1"] for r in rows]
    extra = ""
    if ref is not None:
        extra = f" ({s['spell_f1']-ref['s']['spell_f1']:+.3f}, games better {sum(p>q for p,q in zip(ps,ref['ps']))}/3)"
    print(f"  {tag:>22s}: spellF1 {s['spell_f1']:.3f}{extra} macro {s['macro_f1']:.3f} team {s['team_acc']:.3f} trans {s['trans_f1_25']:.3f} spells {[r['pred_spells'] for r in rows]} per game {[round(x,3) for x in ps]}")
    return s, ps

print("== A. gate radius refinement (team=hidden, as before)")
for r in (0.15, 0.25, 0.35, 0.5, 0.6):
    show(f"gate_r {r}", {"gate_r": r})
print("== B. combined: team=real, gate_r in {0.35, 0.5}")
res = {}
for r in (0.35, 0.5):
    res[r] = show(f"real + gate {r}", {"team": "real", "gate_r": r})
best_r = 0.5 if res[0.5][0]["spell_f1"] >= res[0.35][0]["spell_f1"] else 0.35
cfg0 = {"team": "real", "gate_r": best_r}
s0, ps0 = res[best_r]; ref = {"s": s0, "ps": ps0}
print(f"== C. re-sweep around {cfg0} (reference spellF1 {s0['spell_f1']:.3f})")
for key, vals in {"kappa": [0.005, 0.01, 0.02, 0.05, 0.1], "dead_thr": [0.3, 0.4, 0.5], "shift": [0.0, 0.5, 1.0], "tau": [2.0, 3.0, 5.0], "rate_scale": [0.85, 1.0, 1.15], "min_dead_s": [1.0, 2.0, 4.0]}.items():
    print(f" -- {key}")
    for v in vals:
        show(f"{key}={v}", {**cfg0, key: v}, ref)
