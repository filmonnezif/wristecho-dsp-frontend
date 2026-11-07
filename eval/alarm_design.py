"""Step 1b: redesign the alarm detector on real recordings, using the DESIGN half only.

1. Characterize each real alarm (clean): dominant frequency of its tonal frames (FFT peak,
   1-6 kHz), beep length and gap, and its pattern (T-3, periodic, other).
2. Search detector configurations (probe band x tonality ratio x pattern rule, config.ALARM_GRID)
   on design-half alarms and look-alikes, clean and at 10 / 0 dB SNR. Selection rule, fixed in
   advance: highest alarm recall with ZERO fire alarms on design-half look-alikes; ties go to
   fewer probes (cheaper), then the stricter tonality ratio.
The held-out half is never touched here; eval/real_alarms.py reports on it.
Writes results/alarm_design.json ("chosen" is used by the rest of the pipeline).
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from dsp import frontend  # noqa: E402
from eval import real_alarms as R  # noqa: E402

WIDE = (min(b[0] for b in C.ALARM_GRID["bands"]), max(b[1] for b in C.ALARM_GRID["bands"]))
FREQS = np.arange(WIDE[0], WIDE[1] + 1, C.PROBE_STEP)


def configs():
    g = C.ALARM_GRID
    return [dict(lo=lo, hi=hi, ratio=r, rule=rule) for (lo, hi), r, rule in itertools.product(g["bands"], g["ratios"], g["rules"])]


def characterize(x):
    """Dominant tonal frequency and beep timing of one clean recording."""
    fr = frontend._frames(x) * np.hanning(C.FRAME)
    spec = np.abs(np.fft.rfft(fr, n=1024, axis=1)) ** 2
    f = np.fft.rfftfreq(1024, 1 / C.SR)
    band = (f >= 1000) & (f <= 6000)
    peak_share = spec[:, band].max(1) / (spec.sum(1) + 1e-12)
    tonal = peak_share > 0.2
    if tonal.sum() < 10:
        return dict(f0=None, tonal_share=float(tonal.mean()), pattern="not tonal")
    f0 = float(np.median(f[band][spec[tonal][:, band].argmax(1)]))
    e, sq, total, loud = frontend.probe_energies(x, FREQS)
    on = frontend.tone_on(e, sq, total, loud, FREQS, max(f0 - 300, WIDE[0]), min(f0 + 300, WIDE[1]), 0.15)
    pulses, _ = frontend._pulses(on)
    dur = [b - a for a, b in pulses]
    gaps = [pulses[k + 1][0] - pulses[k][1] for k in range(len(pulses) - 1)]
    if frontend.t3_times(on):
        pattern = "T-3"
    elif frontend.periodic_times(on, 4):
        pattern = "periodic"
    elif dur and max(dur) > 2.0:
        pattern = "continuous tone"
    else:
        pattern = "other"
    return dict(f0=f0, tonal_share=float(tonal.mean()), pattern=pattern, n_pulses=len(pulses),
                beep_s=float(np.median(dur)) if dur else None, gap_s=float(np.median(gaps)) if gaps else None)


def job(clip):
    out = dict(id=clip["id"], group=clip["group"], fire=clip["fire"], fired={})
    for cond, y in R.conditions(clip):
        if cond == "clean" and clip["fire"]:
            out["character"] = characterize(y)
        e, sq, total, loud = frontend.probe_energies(y, FREQS)
        for i, cfg in enumerate(configs()):
            on = frontend.tone_on(e, sq, total, loud, FREQS, cfg["lo"], cfg["hi"], cfg["ratio"])
            out["fired"].setdefault(i, []).append(bool(frontend.decide(on, cfg["rule"])))
    return out


def main():
    if not os.path.exists(R.MANIFEST):
        print("  no data/freesound.json - skipped")
        return
    clips = [c for c in R.clips_downloaded() if c["split"] == "design"]
    with Pool(C.WORKERS) as pool:
        res = [r for r in pool.map(job, clips) if r["fired"]]
    pos, neg = [r for r in res if r["fire"]], [r for r in res if not r["fire"]]
    table = []
    for i, cfg in enumerate(configs()):
        rec = float(np.mean([v for r in pos for v in r["fired"][i]]))
        fa = float(np.mean([v for r in neg for v in r["fired"][i]]))
        table.append(dict(cfg=cfg, recall=rec, fa=fa, probes=len(np.arange(cfg["lo"], cfg["hi"] + 1, C.PROBE_STEP))))
    ok = [t for t in table if t["fa"] == 0]
    best = max(ok, key=lambda t: (round(t["recall"], 6), -t["probes"], t["cfg"]["ratio"]))
    v3 = next(t for t in table if t["cfg"] == C.ALARM_V3)
    chars = [dict(r["character"], id=r["id"], group=r["group"]) for r in pos if "character" in r]
    out = dict(chosen=best["cfg"], chosen_design=best, v3_design=v3, n_design_alarms=len(pos), n_design_lookalikes=len(neg),
               table=table, characterization=chars)
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "alarm_design.json"), "w"), indent=1)

    f0s = [c["f0"] for c in chars if c["f0"]]
    from collections import Counter
    print(f"  design half: {len(pos)} real alarms, {len(neg)} look-alikes (x3 conditions)")
    print(f"  dominant frequency of tonal alarms (Hz): median {np.median(f0s):.0f}, "
          f"in 2.8-3.6 kHz: {np.mean([2800 <= f <= 3600 for f in f0s]):.0%}, range {min(f0s):.0f}-{max(f0s):.0f}")
    print(f"  patterns: {dict(Counter(c['pattern'] for c in chars))}")
    print(f"  v3 detector  recall {v3['recall']:.2f}  look-alike FA {v3['fa']:.2f}")
    print(f"  chosen (v4)  recall {best['recall']:.2f}  look-alike FA {best['fa']:.2f}  {best['cfg']}")
    for t in sorted(table, key=lambda t: -t["recall"])[:8]:
        print(f"    recall {t['recall']:.2f}  FA {t['fa']:.2f}  {t['cfg']}")


if __name__ == "__main__":
    main()
