"""Recall on the predominant-target subset: test events whose FSD50K clip the dataset's raters marked
as present and predominant (data/make_cliplist.py), plus every event not from FSD50K (ESC-50 and
Freesound clips, synthetic alarms). Recomputed from the per-event hits already saved by
eval/evaluate.py, eval/improve.py and eval/repeat.py; nothing is re-run or re-tuned.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from data import streams  # noqa: E402
from eval.extract import DIRS  # noqa: E402


def keep(ev):
    src = ev["source"]
    return src["kind"] != "clip" or src["clip"]["src"] != "fsd" or src["clip"].get("predominant") is True


def main():
    specs = {s["id"]: s for s in streams.specs()}
    order = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(DIRS[False], "test-*.npz")))]
    rows = [(ev["category"], keep(ev)) for sid in order for ev in specs[sid]["events"] if ev["role"] == "target"]
    cat = np.array([c for c, _ in rows])
    sel = np.array([k for _, k in rows])
    print(f"  predominant subset: {sel.sum()} of {len(sel)} test events " +
          ", ".join(f"{c.lower()} {int((sel & (cat == c)).sum())}" for c in C.ORDER))

    def line(name, hits):
        h = np.asarray(hits, bool)
        print(f"  {name:34s} all {h.mean():.3f} -> subset {h[sel].mean():.3f}   " +
              " ".join(f"{c[:5].lower()} {h[sel & (cat == c)].mean():.2f}" for c in C.ORDER))

    r = json.load(open(os.path.join(C.RESULTS, "results.json")))["frontends"]
    for k in ("F0", "F0*", "F0+DSP", "F4 knock@5%", "H0", "H4@5%"):
        if k in r:
            line(k, r[k]["hits"])
    p = os.path.join(C.RESULTS, "improve.json")
    if os.path.exists(p):
        curve = json.load(open(p))["curve"]
        for clf in curve:
            for b, pt in curve[clf].items():
                for mode in ("always-on", "full"):
                    line(f"{clf} @{b} FA/h {mode} (FA/h {pt[mode]['fa_per_hour_events']:.1f})", pt[mode]["hits"])
    p = os.path.join(C.RESULTS, "repeat.json")
    if os.path.exists(p):
        ev = json.load(open(p))["events"]  # in specs order: map to the same subset by stream spec order
        rsel = np.array([keep(e) for sid, s in specs.items() if s["split"] == "test" and s["kind"] == "events"
                         for e in s["events"] if e["role"] == "target"])
        for name, f in (("clean once, stock", lambda e: e["clean_stock"]), ("clean once, best", lambda e: e["clean_best"]),
                        ("repeated, within 8 s", lambda e: e["latency"] is not None and e["latency"] <= 8),
                        ("repeated, within 60 s", lambda e: e["latency"] is not None)):
            v = np.array([f(e) for e in ev], bool)
            print(f"  {name:34s} all {v.mean():.3f} -> subset {v[rsel].mean():.3f}")


if __name__ == "__main__":
    main()
