"""Step 2e: per-class classifier choice, score smoothing, and the false-alarm budget curve.

1. Candidates per category: for each backbone (YAMNet; EfficientAT mn04 / mn10 when cached), its own
   labels' score, its trained head's score, or their mean, each smoothed by a causal moving average
   over the last k = 1, 2 or 3 windows (a one-window spike no longer fires an alert). Each candidate gets thresholds at the false-alarm budget on the tuning
   streams; per category the candidate with the highest tuning recall is kept ("best").
2. Curve: for false-alarm budgets of 0.1-3 per hour, thresholds are re-chosen on the tuning
   streams for stock YAMNet and for "best", and both are measured on the test streams always on
   and as the full system (gate at the 5 % budget + DSP alarm and knock detectors + siren veto).
Nothing here looks at the test streams before the choices are fixed. Writes results/improve.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from eval import evaluate as E  # noqa: E402

SMOOTH = (1, 2, 3)
CURVE_BUDGETS = [0.1, 0.3, 1.0, 3.0]
DATA = []  # set before the worker pools fork, so workers share it


def smooth(x, k):
    """Causal mean over the last k windows."""
    if k == 1:
        return x
    c = np.cumsum(np.r_[np.zeros((1,) + x.shape[1:], x.dtype), x], axis=0)
    i = np.arange(len(x))
    lo = np.maximum(0, i + 1 - k)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None, None]


def backbones():
    from classifier import efficientat, head
    return ["yamnet"] + [m for m in efficientat.MODELS if os.path.exists(head.path_for(m))]


def add_eat(s):
    """Category scores of EfficientAT's own labels and of its trained head, for stream s."""
    from classifier import head, yamnet
    for m in backbones()[1:]:
        z = np.load(os.path.join(C.CACHE, f"eat_{m}", s["id"] + ".npz"))
        sc, emb = z["scores"].astype(np.float32), z["emb"].astype(np.float32)
        s["cat"][f"{m}:stock"] = yamnet.category_scores(sc, head.labels_for(m))
        s["cat"][f"{m}:head"] = head.category_scores(emb, sc, head.load(head.path_for(m)))


def sources():
    return ["stock", "head", "mean"] + [f"{m}:{k}" for m in backbones()[1:] for k in ("stock", "head", "mean")]


def add_variants(s):
    pairs = [("", "stock", "head")] + [(f"{m}:", f"{m}:stock", f"{m}:head") for m in backbones()[1:]]
    for pre, a, b in pairs:
        st, hd = s["cat"][a], s["cat"][b]
        mean = st.copy()
        mean[:, :, 0] = (st[:, :, 0] + hd[:, :, 0]) / 2
        for src, base in ((a, st), (b, hd), (pre + "mean", mean)):
            for k in SMOOTH:
                s["cat"][f"{src}/{k}"] = smooth(base, k)


def tune_split():
    return [s for s in DATA if s["split"] == "tune"]


def candidate(v):
    """Thresholds at the budget, and per category the tuning recall, or -1 if even the strictest
    threshold exceeds the false-alarm budget (an overconfident score cannot be used)."""
    tune = tune_split()
    thr = E.tune_thresholds(tune, v)
    sc = E.score(tune, E.run(dict(kind="F0", clf=v, veto=True, thresholds=thr), tune))
    cats = np.array([r["category"] for r in sc["rows"]])
    h = E.hours(tune, "idle") + E.hours(tune, "events")
    fa = {c: sum(n for (_, cc), n in sc["fa"].items() if cc == c) / h for c in C.ORDER}
    return v, thr, {c: float(sc["hits"][cats == c].mean()) if fa[c] <= C.FA_BUDGET_PER_H + 1e-9 else -1.0
                    for c in C.ORDER}


def point(arg):
    """Thresholds at a false-alarm budget for classifier clf; test results always on and as the full system."""
    clf, budget, cost = arg
    tune = tune_split()
    test = [s for s in DATA if s["split"] == "test"]
    rng = np.random.default_rng(C.SEED)
    thr = E.tune_thresholds(tune, clf, budget)
    on = dict(kind="F0", clf=clf, veto=True, thresholds=thr)
    base = E.score(tune, E.run(on, tune))["hits"]
    table = E.sweep([dict(fe, clf=clf, thresholds=thr) for fe in E.grid("F3", "two-floor+6")], tune, base)
    full = dict(E.choose(table, 0.05)["fe"], kind="F4", alarm="v4", veto=True)
    kn = E.knock_search(tune, full)["chosen"]
    if kn:
        full["knock"] = kn
    out = {}
    for name, fe in (("always-on", on), ("full", full)):
        r = E.summary(fe, test, cost, rng)
        out[name] = dict(recall=r["recall"], recall_ci=r["recall_ci"], recall_snr_ge5=r["recall_snr_ge5"]["recall"],
                         by_category={c: v["recall"] for c, v in r["by_category"].items()},
                         by_snr={k: v["recall"] for k, v in r["by_snr"].items()},
                         fa_per_hour_events=r["fa_per_hour_events"], fa_per_hour_idle=r["fa_per_hour_idle"],
                         duty_idle=r["duty_idle"], duty_events=r["duty_events"], compute_idle=r["compute_idle"],
                         name=r["name"], hits=r["hits"], frontend=fe)
    return clf, budget, thr, out


def main():
    global DATA
    from classifier import head
    from classifier.yamnet import macs_per_window
    from dsp import frontend
    DATA = E.load()
    for s in DATA:
        add_eat(s)
        add_variants(s)
    variants = [f"{src}/{k}" for src in sources() for k in SMOOTH]
    with Pool(min(len(variants), C.WORKERS)) as pool:
        cand = pool.map(candidate, variants)
    print(f"  tuning recall per category at {C.FA_BUDGET_PER_H} FA/h (per candidate):")
    for v, thr, rec in cand:
        print(f"    {v:8s} " + " ".join(f"{c.lower()} {rec[c]:.3f}@{thr[c]:g}" for c in C.ORDER))
    # per category, the best candidate within each set: YAMNet only, one EfficientAT model only (one
    # network on the watch), or any backbone (may need two networks awake)
    sets = {"best_yamnet": lambda v: ":" not in v, "best_mn10": lambda v: v.startswith("mn10_as:"),
            "best_mn04": lambda v: v.startswith("mn04_as:"), "best": lambda v: True}
    choices = {}
    for name, ok in sets.items():
        pool_c = [r for r in cand if ok(r[0])]
        if pool_c:
            choices[name] = {c: max(pool_c, key=lambda r: (r[2][c], -variants.index(r[0])))[0] for c in C.ORDER}
            print(f"  chosen per category ({name}): {choices[name]}", flush=True)
    choice, choice_y = choices["best"], choices["best_yamnet"]
    for s in DATA:
        for name, ch in choices.items():
            s["cat"][name] = np.stack([s["cat"][ch[c]][:, j] for j, c in enumerate(C.ORDER)], axis=1)
    cost = [macs_per_window(), frontend.gate_ops_per_window(), frontend.alarm_ops_per_window(frontend.chosen_alarm()),
            frontend.knock_ops_per_window(), head.macs()]
    jobs = [(clf, b, cost) for clf in ["stock"] + list(choices) for b in CURVE_BUDGETS]
    with Pool(min(len(jobs), C.WORKERS)) as pool:
        pts = pool.map(point, jobs)
    curve = {}
    for clf, b, thr, out in pts:
        curve.setdefault(clf, {})[str(b)] = dict(thresholds=thr, **out)
    tests = {}
    for b in CURVE_BUDGETS:
        sb = curve["stock"][str(b)]
        bb = curve["best"][str(b)]
        for mode in ("always-on", "full"):
            tests[f"best {mode} vs stock always-on @{b}"] = E.mcnemar(np.array(bb[mode]["hits"]), np.array(sb["always-on"]["hits"]))
    json.dump(dict(candidates=[dict(variant=v, thresholds=t, tune_recall=r) for v, t, r in cand], choice=choice,
                   choice_yamnet=choice_y, choices=choices,
                   curve=curve, tests=tests), open(os.path.join(C.RESULTS, "improve.json"), "w"), indent=1)
    print(f"\n  {'classifier':10s} {'FA/h budget':>11s} {'mode':>9s} {'recall':>7s} {'>=5dB':>6s} " +
          " ".join(f"{c[:5].lower():>6s}" for c in C.ORDER) + f" {'FA/h ev':>7s} {'FA/h id':>7s} {'duty id':>7s}")
    for clf in ["stock"] + list(choices):
        for b in CURVE_BUDGETS:
            for mode in ("always-on", "full"):
                r = curve[clf][str(b)][mode]
                print(f"  {clf:10s} {b:11g} {mode:>9s} {r['recall']:7.3f} {r['recall_snr_ge5']:6.3f} " +
                      " ".join(f"{r['by_category'][c]:6.3f}" for c in C.ORDER) +
                      f" {r['fa_per_hour_events']:7.2f} {r['fa_per_hour_idle']:7.2f} {r['duty_idle']:7.3f}")
    for k, v in tests.items():
        print(f"  {k}: p={v['p']:.2g} ({v['only_first']} vs {v['only_second']})")


if __name__ == "__main__":
    main()
