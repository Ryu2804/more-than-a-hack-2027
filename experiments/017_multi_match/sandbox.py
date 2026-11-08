"""Helpers to run code on the marimo/molab sandbox. Sends ONLY code (and reads results back); public datasets are downloaded by the sandbox itself.
Env: MARIMO_TOKEN (required), MARIMO_URL (default: current sandbox), EXEC_SCRIPT (path to marimo-pair's execute-code.sh)."""
import base64, os, re, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
URL = os.environ.get("MARIMO_URL", "https://sb-d6694a8e9e3fd822.sb.molab.run/")
SCRIPT = os.environ.get("EXEC_SCRIPT", "/tmp/claude-1000/-home-ryz-data-dev-competitive-hackathon-More-Than-hack-more-than-a-hack-2027/4e0e52fc-7701-4438-b772-64ecfaeb1ac3/scratchpad/marimo/execute-code.sh")
BUNDLE = {"common.py": EXP / "common.py", "features.py": EXP / "features.py", "other_games.py": EXP / "other_games.py",
          "skillcorner_games.py": EXP / "skillcorner_games.py", "train_model.py": EXP / "015_final_pipeline" / "train_model.py",
          "decoder.py": EXP / "015_final_pipeline" / "decoder.py", "possession_metrics.py": ROOT / "evaluation" / "possession_metrics.py",
          "remote_pipeline.py": HERE / "remote_pipeline.py", "remote_eval.py": HERE / "remote_eval.py"}


def execute(code, timeout=900):
    r = subprocess.run([SCRIPT, "--url", URL, "-"], input=code, capture_output=True, text=True, env=os.environ, timeout=timeout)
    return r.stdout


def common_stub():
    """Only the generic helpers the sandbox code uses (runs, viterbi_2state, viterbi_k, cached, CACHE), cut out of common.py with ast,
    so none of the code that loads the challenge match's tracking is shipped."""
    import ast
    src = (EXP / "common.py").read_text(); tree = ast.parse(src)
    keep = {"runs", "viterbi_2state", "viterbi_k", "cached"}
    parts = ["import os\nimport numpy as np\nCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_cache')\n"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in keep:
            parts.append(ast.get_source_segment(src, node))
    return "\n\n".join(parts) + "\n"


def ship(workdir):
    files = {k: v.read_text() for k, v in BUNDLE.items() if k != "common.py"}
    files["common.py"] = common_stub()
    assert not any(m in t for t in files.values() for m in ("atd_raw", "atd_full", "BAR-ATH_tactical")), "match-data reference in payload"
    code = f"import pathlib\nW=pathlib.Path('{workdir}'); W.mkdir(parents=True, exist_ok=True)\nfiles={files!r}\nfor k,v in files.items(): (W/k).write_text(v)\nprint('shipped',len(files),'files to',W)\n"
    return execute(code).strip().splitlines()[-1]


def run_bg(workdir, cmd, log="job.log"):
    code = f"import subprocess, sys, pathlib\nW=pathlib.Path('{workdir}'); lf=open(W/'{log}','w')\np=subprocess.Popen({cmd!r}, shell=True, cwd=str(W), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)\nprint('launched',p.pid)\n"
    return execute(code).strip().splitlines()[-1]


def tail(workdir, log="job.log", n=800):
    return execute(f"import os; p='{workdir}/{log}'; print(open(p).read()[-{n}:] if os.path.exists(p) else 'NOLOG')")


def wait(workdir, marker, log="job.log", timeout=3600, every=20, verbose=True):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = tail(workdir, log, 1500)
        if marker in out:
            return True, out
        if "Traceback" in out:
            return False, out
        time.sleep(every)
    return False, "timeout"


def get_bytes(path, chunk=450000):
    size = int(re.findall(r"\d+", execute(f"import os; print('SIZE', os.path.getsize('{path}'))").split("SIZE")[-1])[0])
    nchunks = size * 4 // 3 // chunk + 2
    buf = ""
    for c in range(nchunks):
        out = execute(f"import base64,sys; s=base64.b64encode(open('{path}','rb').read()).decode(); n={chunk}; sys.stdout.write('<<'+s[{c}*n:({c}+1)*n]+'>>')")
        buf += re.search(r"<<([A-Za-z0-9+/=]*)>>", out).group(1)
    return base64.b64decode(buf)
