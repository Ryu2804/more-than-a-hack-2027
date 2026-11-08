"""Run a training configuration on the marimo/molab sandbox and score it locally. NO local training.

    MARIMO_TOKEN=... python remote.py NAME [train_model.py args...]      e.g.  remote.py seeds8 --seeds 8

Sends ONLY code (features.py, other_games.py, train_model.py: no data of the challenge match); the sandbox downloads the public
Metrica games itself, trains the held-out folds and writes fold_game*.npz, which are copied back to experiments/_cache/folds_NAME.
Needs MARIMO_TOKEN (env) and MARIMO_URL (env, default: the current molab sandbox).
"""
import base64, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
URL = os.environ.get("MARIMO_URL", "https://sb-d6694a8e9e3fd822.sb.molab.run/")
SCRIPT = Path(os.environ.get("EXEC_SCRIPT", "/tmp/claude-1000/-home-ryz-data-dev-competitive-hackathon-More-Than-hack-more-than-a-hack-2027/4e0e52fc-7701-4438-b772-64ecfaeb1ac3/scratchpad/marimo/execute-code.sh"))


def execute(code):
    r = subprocess.run([str(SCRIPT), "--url", URL, "-"], input=code, capture_output=True, text=True, env=os.environ, timeout=600)
    return r.stdout


def launch(name, args):
    files = {"features.py": (EXP / "features.py").read_text(), "other_games.py": (EXP / "other_games.py").read_text(),
             "train_model.py": (EXP / "015_final_pipeline" / "train_model.py").read_text(),
             "common.py": 'import os\nCACHE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_cache")\n'}
    assert not any(k in v for v in files.values() for k in ("BAR-ATH", "atd_raw", "atd_full")), "match data reference in payload"
    code = (f"import os, subprocess, sys, pathlib, shutil\nW=pathlib.Path('/tmp/work_{name}'); shutil.rmtree(W, ignore_errors=True); W.mkdir()\n"
            f"files={files!r}\nfor k,v in files.items(): (W/k).write_text(v)\n(W/'out').mkdir()\nlog=open(W/'train.log','w')\n"
            f"p=subprocess.Popen([sys.executable,'-u',str(W/'train_model.py'),'--out',str(W/'out'),'--folds']+{args!r},cwd=str(W),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint('launched',p.pid)\n")
    print(name, execute(code).strip().splitlines()[-1:])


def wait(name, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = execute(f"import os; p='/tmp/work_{name}/train.log'; print(open(p).read()[-400:] if os.path.exists(p) else 'NOLOG')")
        if "folds done" in out:
            return True
        if "Traceback" in out or "Error" in out:
            print(out[-600:]); return False
        time.sleep(15)
    return False


def fetch(name):
    dest = EXP / "_cache" / f"folds_{name}"; dest.mkdir(parents=True, exist_ok=True)
    for g in (1, 2, 3):
        buf = ""
        for c in range(4):
            out = execute(f"import base64,sys; s=base64.b64encode(open('/tmp/work_{name}/out/fold_game{g}.npz','rb').read()).decode(); n=450000; sys.stdout.write('<<'+s[{c}*n:({c}+1)*n]+'>>')")
            buf += re.search(r"<<([A-Za-z0-9+/=]*)>>", out).group(1)
        (dest / f"fold_game{g}.npz").write_bytes(base64.b64decode(buf))
    return dest


if __name__ == "__main__":
    name, args = sys.argv[1], sys.argv[2:]
    launch(name, args)
    ok = wait(name)
    print("trained" if ok else "FAILED")
    if ok:
        print("fetched to", fetch(name))
