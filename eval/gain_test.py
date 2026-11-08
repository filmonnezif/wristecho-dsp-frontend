"""Check: does a fixed input gain help the final system on the full benchmark? (demo/experiments/gainSirenTest.m
found +12 dB raised siren recall from 6 to 10 of 24 sirens in MATLAB.)

Every test stream (tuning-free: the paper's final system, best_mn10, at budgets 0.1 and 1 false alarms per hour per
class; choice, thresholds and gate settings from results/improve.json, unchanged) is rendered, multiplied by the gain,
clipped to [-1, 1], and run through the whole system: gate, Goertzel/T-3 alarm detector, EfficientAT-mn10 + head,
alert rule. Reported per gain: recall (all, per class), false alarms per hour (event and idle streams), classifier
duty, clipped samples, and exact McNemar tests against no gain. Writes results/gain_test.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import efficientat, yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402
from eval import evaluate as E  # noqa: E402
from eval import real_scenes as RS  # noqa: E402
from eval.final import MODEL, settings  # noqa: E402
from eval.improve import smooth  # noqa: E402

GAINS = [0, 6, 12]
BUDGETS = ["0.1", "1.0"]
SET = None


def prep(spec):
    """CPU worker: the stream at each gain, its gate masks (per budget) and DSP alarm times."""
    x, events = streams.render(spec)
    cfg = frontend.chosen_alarm()
    out = []
    for g in GAINS:
        y = x * 10 ** (g / 20)
        clipped = int(np.sum(np.abs(y) > 1))
        y = np.clip(y, -1, 1).astype(np.float32)
        _, bands = frontend.band_power(y)
        masks = {b: RS.gate_mask(y, SET[b]["fe"], bands) for b in BUDGETS}
        alarms = [(t, "FIRE_ALARM") for t in frontend.alarm_times(y, cfg=cfg)]
        out.append((g, y, clipped, masks, alarms))
    return spec, events, out


def classify(y, s):
    """Category scores of the final classifier (mn10 + head, per-class source and smoothing) for audio y."""
    from classifier import head
    scores, emb = efficientat.window_outputs(y, MODEL)
    h = head.load(head.path_for(MODEL))
    base = {"stock": yamnet.category_scores(scores, head.labels_for(MODEL)), "head": head.category_scores(emb, scores, h)}
    base["mean"] = base["stock"].copy()
    base["mean"][:, :, 0] = (base["stock"][:, :, 0] + base["head"][:, :, 0]) / 2
    cat = np.zeros_like(base["head"])
    for j, c in enumerate(C.ORDER):
        src, k = s["choice"][c].split(":")[1].split("/")
        cat[:, j] = smooth(base[src], int(k))[:, j]
    return cat


def main():
    global SET
    SET = {b: settings("best_mn10", b) for b in BUDGETS}
    specs = [s for s in streams.specs() if s["split"] == "test"]
    res = {(g, b): [] for g in GAINS for b in BUDGETS}
    rows = {(g, b): [] for g in GAINS for b in BUDGETS}
    clip = {g: [0, 0] for g in GAINS}
    with Pool(C.WORKERS) as pool:
        for i, (spec, events, out) in enumerate(pool.imap_unordered(prep, specs), 1):
            for g, y, clipped, masks, alarms in out:
                clip[g][0] += clipped
                clip[g][1] += len(y)
                cat = classify(y, SET[BUDGETS[0]])  # the score choice is the same at every budget
                for b in BUDGETS:
                    m = masks[b]
                    fired = yamnet.detect(cat, m, alarms, SET[b]["thresholds"], siren_veto=True)
                    s = dict(spec, events=events, n=len(cat))
                    rows[(g, b)].append(s)
                    res[(g, b)].append((m, fired))
            if i % 50 == 0:
                print(f"  {i}/{len(specs)} streams", flush=True)
    out = {}
    base_hits = {}
    for b in BUDGETS:
        for g in GAINS:
            order = sorted(range(len(rows[(g, b)])), key=lambda k: rows[(g, b)][k]["id"])
            ss = [rows[(g, b)][k] for k in order]
            rr = [res[(g, b)][k] for k in order]
            sc = E.score(ss, rr)
            cats = np.array([r["category"] for r in sc["rows"]])
            h_idle, h_ev = E.hours(ss, "idle"), E.hours(ss, "events")
            awake = [m for (m, _), s in zip(rr, ss)]
            r = dict(recall=float(sc["hits"].mean()), by_category={c: float(sc["hits"][cats == c].mean()) for c in C.ORDER},
                     siren_hits=int(sc["hits"][cats == "SIREN"].sum()),
                     fa_per_hour_events=sum(v for (k, _), v in sc["fa"].items() if k == "events") / h_ev,
                     fa_per_hour_idle=sum(v for (k, _), v in sc["fa"].items() if k == "idle") / h_idle,
                     classifier_calls=int(sum(int(m.sum()) for m in awake)),
                     duty_idle=float(np.mean([m.mean() for m, s in zip(awake, ss) if s["kind"] == "idle"])),
                     clipped_share=clip[g][0] / clip[g][1])
            if g == 0:
                base_hits[b] = (sc["hits"], cats)
            else:
                h0, c0 = base_hits[b]
                r["mcnemar_all"] = E.mcnemar(sc["hits"], h0)
                r["mcnemar_siren"] = E.mcnemar(sc["hits"][cats == "SIREN"], h0[c0 == "SIREN"])
            out[f"{g}@{b}"] = r
    json.dump(out, open(os.path.join(C.RESULTS, "gain_test.json"), "w"), indent=1)
    print(f"\n  {'gain':>5s} {'budget':>6s} {'recall':>7s} " + " ".join(f"{c[:5].lower():>6s}" for c in C.ORDER) +
          f" {'sirens':>7s} {'FA/h ev':>8s} {'FA/h id':>8s} {'calls':>7s} {'duty id':>8s} {'clipped':>9s}  vs 0 dB (all; siren)")
    for b in BUDGETS:
        for g in GAINS:
            r = out[f"{g}@{b}"]
            t = (f"+{r['mcnemar_all']['only_first']}/-{r['mcnemar_all']['only_second']} p={r['mcnemar_all']['p']:.2g}; "
                 f"+{r['mcnemar_siren']['only_first']}/-{r['mcnemar_siren']['only_second']} p={r['mcnemar_siren']['p']:.2g}") if g else ""
            print(f"  {g:+4d}dB {b:>6s} {r['recall']:7.3f} " + " ".join(f"{r['by_category'][c]:6.3f}" for c in C.ORDER) +
                  f" {r['siren_hits']:4d}/96 {r['fa_per_hour_events']:8.2f} {r['fa_per_hour_idle']:8.2f} {r['classifier_calls']:7d}"
                  f" {r['duty_idle']:8.3f} {100 * r['clipped_share']:8.4f}%  {t}")


if __name__ == "__main__":
    main()
