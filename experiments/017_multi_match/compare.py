"""Score variant prediction folders on the sandbox (remote_eval.py) and print one comparison table.

    MARIMO_TOKEN=... python compare.py full sk_only metrica_only ...            (variants live in /tmp/work_multi/<name>/out/preds)
Options: --configs '[{...}, ...]' (default: the current decoder with a 0.5 s merge).
"""
import argparse, json, sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
import sandbox as sb

W = "/tmp/work_multi"
ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--configs", default='[{"merge":0.5}]')
a = ap.parse_args()
cfgs = json.loads(a.configs)
data = {}
for name in a.names:
    out = f"{W}/{name}/eval.json"
    sb.run_bg(W, f"python -u remote_eval.py --games-dir {W}/games --preds-dir {W}/{name}/out/preds --configs '{a.configs}' --out {out}", f"eval_{name}.log")
for name in a.names:
    out = f"{W}/{name}/eval.json"
    t0 = time.time()
    while time.time() - t0 < 2400:
        if sb.execute(f"import os; print('YES' if os.path.exists('{out}') and os.path.getsize('{out}')>100 else 'NO')").strip().endswith("YES"):
            break
        time.sleep(10)
    data[name] = json.loads(sb.get_bytes(out).decode())
def agg(res, key, which, ci, src):
    v = [(r["baseline"] if which == "base" else r["configs"][ci])[key] for r in res if r["game"].startswith(src)]
    return float(np.mean(v)), v
print(f'{"variant":18s} {"cfg":3s} | {"Metrica (3)":>26s} | {"SkillCorner (20)":>26s} | {"all 23":>7s} | per-source spellF1 baseline')
for name in a.names:
    res = data[name]["results"]
    for ci in range(len(cfgs)):
        mm, _ = agg(res, "spell_f1", "cfg", ci, "metrica"); sk, skv = agg(res, "spell_f1", "cfg", ci, "skillcorner")
        mmM, _ = agg(res, "macro_f1", "cfg", ci, "metrica"); skM, _ = agg(res, "macro_f1", "cfg", ci, "skillcorner")
        bm, _ = agg(res, "spell_f1", "base", 0, "metrica"); bs, _ = agg(res, "spell_f1", "base", 0, "skillcorner")
        allv = [r["configs"][ci]["spell_f1"] for r in res]
        print(f'{name:18s} {ci:3d} | spell {mm:.3f} macro {mmM:.3f}      | spell {sk:.3f} macro {skM:.3f}      | {np.mean(allv):.3f}  | base M {bm:.3f} SC {bs:.3f}')
json.dump(data, open(f"/tmp/claude-1000/-home-ryz-data-dev-competitive-hackathon-More-Than-hack-more-than-a-hack-2027/4e0e52fc-7701-4438-b772-64ecfaeb1ac3/scratchpad/compare_last.json", "w"))
