"""Run Rasya's deployable decoder (origin/rasya: touch-hold ownership + out_or_slow2 dead cue + merge k picked on the other matches) on the 7 IDSSE matches
leave-one-match-out, and score it with the official evaluate() (same scorer as our harness). Sandbox job; public IDSSE data only.
Usage: python rasya_on_idsse.py OUTJSON   (expects /tmp/rasya_cmp/{experiments,evaluation} and parsed npz in /tmp/idsse/parsed)"""
import json, os, sys
import numpy as np
R = "/tmp/rasya_cmp"
for d in ("012_spellf1", "011_ball_value", "007_idsee"):
    sys.path.insert(0, f"{R}/experiments/{d}")
sys.path.insert(0, f"{R}/evaluation")
import harness as H
H.PARSED = "/tmp/idsse/parsed"; H.load.__defaults__ = ("/tmp/idsse/parsed",)
import phase2 as P2, touch as T, deployable as D
import possession_metrics as pm
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
NAMES = np.array(["H", "A", "D", "X"])
ctx = {}
for m in MATCHES:
    c = T.match_context(m); npz = H.load(m)
    c["cue"], _, _ = P2.cue_masks(npz, npz["ball_x"], npz["ball_y"])
    c["smooth50"] = H.smooth_team(c["team"], 50); c["held"] = T.hold(c["dmin"], c["team"], D.R_M, D.MINLEN)
    ctx[m] = c
def ev(c, pred):
    r = pm.evaluate(pm.label_codes(NAMES[c["gt"]]), pm.label_codes(NAMES[pred]), bootstrap_n=0)
    return dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], pred_spells=r["n_pred_spells"], gt_spells=r["n_gt_spells"], alive_dead_f1=r["alive_dead_f1"], team_acc=r["team_acc_both_alive"])
out = {}
for m in MATCHES:
    train = [t for t in MATCHES if t != m]
    def mean_for(k, arm):
        return float(np.mean([D.f1_of(ctx[t], D.build_prediction(ctx[t], ctx[t]["cue"], ctx[t]["held"] if arm == "touch" else ctx[t]["smooth50"], k))[0] for t in train]))
    best_k = max(D.KS, key=lambda k: (mean_for(k, "touch"), -k))
    c = ctx[m]
    out[m] = dict(k=best_k, flipped=bool(c["flipped"]),
                  rasya_touch=ev(c, D.build_prediction(c, c["cue"], c["held"], best_k)),
                  rasya_nearest50=ev(c, D.build_prediction(c, c["cue"], c["smooth50"], best_k)),
                  rasya_touch_oracle_dead=ev(c, D.build_prediction(c, c["cue"], c["held"], 0, oracle_dead=True)))
    print(m, best_k, {k: round(v["spell_f1"], 3) for k, v in out[m].items() if isinstance(v, dict)}, flush=True)
json.dump(out, open(sys.argv[1], "w"))
print("RASYA DONE", flush=True)
