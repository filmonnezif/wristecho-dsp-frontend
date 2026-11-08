"""Collect the numbers the paper uses into one small file: paper/data/results_v7.json (no per-event
hits, no calibration tables). Run after eval/improve.py and eval/final.py."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from data import streams  # noqa: E402
from eval.extract import DIRS  # noqa: E402
from eval.predominant import keep  # noqa: E402

OUT = os.path.join(C.ROOT, "paper", "data", "results_v7.json")


def main():
    specs = {s["id"]: s for s in streams.specs()}
    order = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(DIRS[False], "test-*.npz")))]
    rows = [(ev["category"], keep(ev)) for sid in order for ev in specs[sid]["events"] if ev["role"] == "target"]
    cat = np.array([c for c, _ in rows])
    sel = np.array([k for _, k in rows])
    imp = json.load(open(os.path.join(C.RESULTS, "improve.json")))
    curve = {}
    for clf, pts in imp["curve"].items():
        for b, pt in pts.items():
            for mode in ("always-on", "full"):
                r = pt[mode]
                h = np.array(r["hits"], bool)
                curve.setdefault(clf, {}).setdefault(b, {})[mode] = dict(
                    {k: v for k, v in r.items() if k not in ("hits", "frontend")},
                    subset=float(h[sel].mean()), subset_by_category={c: float(h[sel & (cat == c)].mean()) for c in C.ORDER})
            curve[clf][b]["thresholds"] = pt["thresholds"]
            curve[clf][b]["frontend"] = pt["full"].get("frontend")
    res = json.load(open(os.path.join(C.RESULTS, "results.json")))
    f = res["frontends"]
    heads = {}
    for p in glob.glob(os.path.join(C.RESULTS, "head*.json")):
        h = json.load(open(p))
        heads[os.path.basename(p)[:-5]] = dict(chosen=h["chosen"], test_ap_head=h["test_ap_head"], test_ap_stock=h["test_ap_stock"],
                                               train_counts=h["train_counts"],
                                               selection=[dict(kind=s["kind"], params=s["params"], tune_map=s["tune_map"])
                                                          for s in h["selection"]])
    extra = {}
    for name in ("freesound_train.json", "us8k_train.json"):
        p = os.path.join(C.ROOT, "data", name)
        if os.path.exists(p):
            from collections import Counter
            extra[name] = dict(Counter(c["category"] for c in json.load(open(p))))
    out = dict(
        subset=dict(n=int(sel.sum()), of=len(sel), by_category={c: int((sel & (cat == c)).sum()) for c in C.ORDER}),
        choices=imp.get("choices"), tests=imp.get("tests"), curve=curve, heads=heads, extra_training=extra,
        app={k: {x: f[k][x] for x in ("recall", "recall_ci", "by_category", "fa_per_hour_events", "fa_per_hour_idle", "duty_idle",
                                       "by_snr", "latency_median_s")} for k in ("F0", "F0*", "F0+DSP", "F3@5%", "F4@5%", "F4 knock@5%")
             if k in f},
        gate=dict(selected_variant=res["selected_variant"], ablation=res["ablation"], ops=res["ops_per_window"],
                  pareto_test=res.get("pareto_test"), duty_idle_by_env={k: f[k]["duty_idle_by_env"] for k in ("F2@5%", "F3@5%") if k in f}),
        hours=res["hours"], n_test_targets=res["n_test_targets"],
        final=json.load(open(os.path.join(C.RESULTS, "final.json"))) if os.path.exists(os.path.join(C.RESULTS, "final.json")) else None,
        alarm_design=json.load(open(os.path.join(C.RESULTS, "alarm_design.json")))["chosen"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"  {OUT}: {os.path.getsize(OUT):,} bytes")


if __name__ == "__main__":
    main()
