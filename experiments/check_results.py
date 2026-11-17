"""Check each experiment's headline numbers against its committed evidence.

No match data, caches or models are needed: numbers are recomputed from each experiment's
`results/*.json`, from the committed baseline CSVs and from the public ground truth. The expected
values are the ones quoted in `EXX_*/CARD.md` and `REGISTRY.md`.

    python experiments/check_results.py            # every experiment
    python experiments/check_results.py E06 E07    # selected experiments

Exit status 0 means every check passed.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

LOCKED_CSV = "cd59fb902882f34d53983cf78478b93a3757bafde65d83e5502f157be214fdb8"


def load(experiment, name):
    return json.loads((HERE / experiment / "results" / name).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_gt():
    return pm.label_codes(pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))


def e00():
    gt = public_gt()
    runs = pm.valid_runs(gt)
    graded = sum(b - a for a, b in runs)
    spells = pm.possession_spells(gt, runs)
    counts = {k: int((gt == i).sum()) for i, k in enumerate("HAD")}
    return [("public GT graded frames", graded, 7500), ("public GT possessions", len(spells), 17),
            ("public GT H frames", counts["H"], 3454), ("public GT A frames", counts["A"], 2577),
            ("public GT D frames", counts["D"], 1469)]


def e01():
    gt = public_gt()
    expected = {"raw": 0.037838, "smooth25": 0.286792, "smooth50": 0.325581, "smooth125": 0.424242}
    out = []
    for name, value in expected.items():
        pred = pm.label_codes(pm.read_label_csv(ROOT / f"evaluation/results/baseline_{name}.csv"))
        out.append((f"public spellF1 baseline_{name}", pm.evaluate(gt, pred, taus=pm.TAUS, bootstrap_n=0)["spell_f1_mean"], value))
    return out


def _selected_record(data):
    return next(r for r in data["records"] if r["method"] == data["selected"]["method"]
                and r["config"]["name"] == data["selected"]["config"]["name"])


def e02():
    rec = _selected_record(load("E02_wave1", "wave1_metrics.json"))
    return [("selected m01_hysteresis/r15", f"{rec['method']}/{rec['config']['name']}", "m01_hysteresis/r15"),
            ("external mean spellF1", rec["external_mean_spellF1"], 0.460517)]


def e03():
    rec = _selected_record(load("E03_wave2", "wave2_metrics.json"))
    return [("selected w08_symmetric_boost/cost6", f"{rec['method']}/{rec['config']['name']}", "w08_symmetric_boost/cost6"),
            ("external mean spellF1", rec["external_mean_spellF1"], 0.517438)]


def e04():
    dev = load("E04_wave3", "wave3_development.json")
    best = max(dev["records"], key=lambda r: r["external_mean_spellF1"])
    pub = next(r for r in load("E04_wave3", "wave3_public.json")["records"] if r["candidate"] == "selected")
    return [("selected family/decoder", f"{dev['selected']['family']}/{dev['selected']['decoder']['name']}", "hybrid_tree25/centered2"),
            ("best development spellF1", best["external_mean_spellF1"], 0.615394),
            ("configurations", len(dev["records"]), 66),
            ("public diagnostic spellF1", pub["spellF1"], 0.575610)]


def _family_decoder(record):
    return record["family"], record["decoder"]["name"]


def e05():
    dev = load("E05_wave4", "wave4_development.json")
    sel = (dev["selected"]["family"], dev["selected"]["decoder"]["name"])
    rec = next(r for r in dev["records"] if _family_decoder(r) == sel)
    pub = next(r for r in load("E05_wave4", "wave4_public.json")["records"] if r["candidate"] == "selected")
    return [("selected family/decoder", "/".join(sel), "pp_gru25/no_shift"),
            ("development equal-provider spellF1", rec["external_mean_spellF1"], 0.656490),
            ("configurations", len(dev["records"]), 64),
            ("public diagnostic spellF1", pub["spellF1"], 0.772414)]


def e06():
    dev = load("E06_wave5", "wave5_development.json")
    rec = {_family_decoder(r): r for r in dev["records"]}
    sel, ctl = rec[("short_gru25", "dead45")], rec[("old_gru25", "wave4")]
    pub = next(r for r in load("E06_wave5", "wave5_public.json")["records"] if r["candidate"] == "selected")
    holdout = load("E06_wave5", "wave5_holdout.json")
    return [("selected family/decoder", f"{dev['selected']['family']}/{dev['selected']['decoder']['name']}", "short_gru25/dead45"),
            ("promoted", dev["promoted"], True),
            ("eligible configurations", sum(r["eligible"] for r in dev["records"]), 6),
            ("development equal-provider spellF1", sel["external_mean_spellF1"], 0.670099),
            ("control (wave4) spellF1", ctl["external_mean_spellF1"], 0.656490),
            ("gain vs control", sel["gain_vs_control"], 0.013609),
            ("public diagnostic spellF1", pub["spellF1"], 0.772414),
            ("public diagnostic macroF1", pub["macroF1"], 0.902059),
            ("public audit CSV hash = locked", pub["sha256"], LOCKED_CSV),
            ("independent rebuild hash = locked", holdout["reproduced_sha256"], LOCKED_CSV),
            ("root submission.csv hash = locked", sha256(ROOT / "submission.csv"), LOCKED_CSV),
            ("submission/artifacts CSV hash = locked", sha256(ROOT / "submission/artifacts/farras_wave5.csv"), LOCKED_CSV)]


def e07():
    dev = load("E07_wave6", "wave6_development.json")
    best = next(r for r in dev["records"] if r["name"] == dev["best_new"])
    pub = next(r for r in load("E07_wave6", "wave6_public.json")["records"] if r["candidate"] == "best_new")
    return [("best alternative", dev["best_new"], "m05_dead_hysteresis/low35_high50"),
            ("best development spellF1", best["external_mean_spellF1"], 0.674807),
            ("gain vs wave5", best["gain_vs_control"], 0.004707),
            ("gain below 0.005 promotion threshold", best["gain_vs_control"] < 0.005, True),
            ("promoted", dev["promoted"], False),
            ("bootstrap 95% lower", best["paired_bootstrap"]["lower"], 0.003412),
            ("public diagnostic spellF1 (not adopted)", pub["spellF1"], 0.786207)]


def e08():
    rows = {r["family"]: r["mean_spellF1"] for r in load("E08_fair_idsse", "fair_idsse_results.json")["summary"]}
    return [("short-loss PP-TCN", rows["short_tcn"], 0.595755), ("SetSeq", rows["setseq"], 0.578602),
            ("PP-TCN", rows["pp_tcn"], 0.571090), ("HGB symmetric", rows["hgb"], 0.315719)]


def _research(experiment, name, primary, delta, promoted, extra=()):
    data = load(experiment, name)
    p = data["primary"]
    out = [("primary mean spellF1", p["mean_spellF1"], primary),
           ("primary delta vs control", p["paired_vs_control"]["mean_delta"], delta),
           ("primary promoted for research", data["primary_promoted_for_research"], promoted)]
    records = {r.get("key") or f"{r['family']}/{r['decoder']}": r for r in data.get("records", [])}
    for label, key, value in extra:
        out.append((label, records[key]["mean_spellF1"], value))
    return out


def e09():
    return _research("E09_cross_branch", "cross_branch_results.json", 0.609030, 0.013274, True,
                     [("best raw short75+SetSeq25/dead45", "short_setseq25/dead45", 0.615458)])


def e10():
    return _research("E10_temporal_methods", "temporal_methods_results.json", 0.605838, -0.009620, False,
                     [("best exploratory consensus", "consensus/alpha0.75_contact0.5", 0.616970)])


def e11():
    return _research("E11_spell_training", "spell_training_results.json", 0.618941, 0.003484, False,
                     [("best exploratory event_replace25/consensus", "event_replace25/consensus", 0.619563)])


def e12():
    return _research("E12_event_learning", "event_learning_results.json", 0.619192, -0.000371, False,
                     [("best exploratory contact_replace/consensus", "contact_replace/consensus", 0.624577),
                      ("ranking_replace/dead45", "ranking_replace/dead45", 0.621512)])


def e13():
    data = load("E13_provider_generalization", "provider_generalization_results.json")
    fam = {r["family"]: r["summary"]["clean"] for r in data["records"]}
    return [("control equal-provider spellF1", fam["control"]["equal_provider_spellF1"], 0.540864),
            ("primary modality_dropout", fam["modality_dropout"]["equal_provider_spellF1"], 0.530378),
            ("primary delta vs control", fam["modality_dropout"]["paired_vs_control"]["mean_delta"], -0.010486),
            ("RDrop", fam["rdrop"]["equal_provider_spellF1"], 0.540882),
            ("primary promoted for research", data["primary_promoted_for_research"], False),
            ("unique target matches", data["unique_target_matches"], 27)]


CHECKS = {"E00": e00, "E01": e01, "E02": e02, "E03": e03, "E04": e04, "E05": e05, "E06": e06, "E07": e07,
          "E08": e08, "E09": e09, "E10": e10, "E11": e11, "E12": e12, "E13": e13}


def same(actual, expected):
    if isinstance(expected, float) and not isinstance(expected, bool):
        return round(float(actual), 6) == round(expected, 6)
    return actual == expected


def main(argv):
    selected = [a.upper()[:3] for a in argv] or list(CHECKS)
    failures = 0
    for key in selected:
        print(f"== {key}")
        for label, actual, expected in CHECKS[key]():
            ok = same(actual, expected)
            failures += not ok
            shown = f"{actual:.6f}" if isinstance(actual, float) else str(actual)[:16]
            print(f"  {'PASS' if ok else 'FAIL'}  {label}: {shown}" + ("" if ok else f" (expected {expected})"))
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
