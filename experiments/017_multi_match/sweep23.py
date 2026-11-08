"""Decoder sweep on the 23 held-out matches (runs remote_eval.py on the sandbox). Usage: sweep23.py [name]  (preds in /tmp/work_multi/<name>/preds)."""
import json, sys, time
import numpy as np
import sandbox as sb
name = sys.argv[1] if len(sys.argv) > 1 else "full"
W = "/tmp/work_multi"
base = {"merge": 0.5}
cfgs = [base]
for k, vs in {"merge": [0.0, 0.25, 1.0, 1.5], "gate_r": [0.25, 0.5, 0.75], "kappa": [0.0, 0.005, 0.05, 0.1], "dead_thr": [0.4, 0.6], "shift": [0.5, 1.0],
              "tau": [2.0, 5.0], "rate_scale": [0.7, 0.85, 1.15, 1.3], "min_dead_s": [1.0, 3.0, 4.0]}.items():
    cfgs += [{**base, k: v} for v in vs]
out = f"{W}/{name}/sweep23.json"
print(len(cfgs), "configs", sb.run_bg(f"{W}/{name}", f"python -u remote_eval.py --games-dir {W}/games --preds-dir {W}/{name}/preds --configs '{json.dumps(cfgs)}' --out {out}", f"sweep_{name}.log"))
t0 = time.time()
while time.time() - t0 < 1800:
    if sb.execute(f"import os; print('YES' if os.path.exists('{out}') and os.path.getsize('{out}')>100 else 'NO')").strip().endswith("YES"): break
    log = sb.tail(f"{W}/{name}", f"sweep_{name}.log", 400)
    if "Traceback" in log: print(log); sys.exit(1)
    time.sleep(10)
d = json.loads(sb.get_bytes(out).decode()); json.dump(d, open(f"/tmp/claude-1000/-home-ryz-data-dev-competitive-hackathon-More-Than-hack-more-than-a-hack-2027/4e0e52fc-7701-4438-b772-64ecfaeb1ac3/scratchpad/sweep23_{name}.json", "w"))
res = d["results"]; ref = [r["configs"][0]["spell_f1"] for r in res]
print(f"baseline(heuristic) spell {np.mean([r['baseline']['spell_f1'] for r in res]):.3f}; ref cfg {cfgs[0]} spell {np.mean(ref):.3f} macro {np.mean([r['configs'][0]['macro_f1'] for r in res]):.3f}")
for i, c in enumerate(cfgs[1:], 1):
    v = [r["configs"][i]["spell_f1"] for r in res]
    print(f"{json.dumps({k: c[k] for k in c if c[k] != base.get(k)}):28s} spell {np.mean(v):.3f} ({np.mean(v)-np.mean(ref):+.3f}) wins {sum(a>b for a,b in zip(v,ref))}/23 ties {sum(a==b for a,b in zip(v,ref))} pred_spells {np.mean([r['configs'][i]['pred_spells'] for r in res]):.0f} gt {np.mean([r['configs'][i]['gt_spells'] for r in res]):.0f}")
