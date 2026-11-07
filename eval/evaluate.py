"""Step 2 (fast): evaluate every front end as a mask over the cached classifier scores.

Front ends
  F0   classifier on every window (the deployed baseline)
  F1   classifier on every k-th window (naive duty cycling)
  F2   fixed energy gate: wake when the HPF'd level exceeds T dBFS
  F3   adaptive gate: wake when a band rises Delta dB above its tracked noise floor
  F4   F3 + the Goertzel/T-3 alarm detector, which raises fire alarms by itself (proposed)
  F0+A classifier on every window + the alarm detector (ablation: separates the two effects)
Gates keep the classifier awake for a hangover H after the last trigger.

Calibration (tuning environments only): for each gate, the setting with the lowest classifier
duty such that at most MISS_BUDGET of the events F0 detects are lost because the gate slept.
Everything reported is then measured on the test environments, never seen while tuning.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier.yamnet import detect  # noqa: E402


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(C.CACHE, "*.npz"))):
        z = np.load(p)
        out.append(dict(cat=z["cat"].astype(np.float32), level=z["level"], novelty=z["novelty"],
                        alarms=[(float(t), "FIRE_ALARM") for t in z["alarms"]], cost=z["cost"],
                        **json.loads(str(z["meta"]))))
    return out


def hang(trig, h_s):
    n = int(round(h_s / (C.HOP / C.SR)))
    return np.convolve(trig.astype(int), np.ones(n + 1, int))[:len(trig)] > 0


def mask(fe, s):
    kind = fe[0]
    if kind in ("F0", "F0+A"):
        return None
    if kind == "F1":
        return np.arange(len(s["cat"])) % fe[1] == 0
    if kind == "F2":
        return hang(s["level"] > fe[1], fe[2])
    return hang(s["novelty"] > fe[1], fe[2])  # F3, F4


def run(fe, streams):
    """Per stream: (awake mask or None, detections)."""
    res = []
    for s in streams:
        m = mask(fe, s)
        extra = s["alarms"] if fe[0] in ("F4", "F0+A") else ()
        res.append((m, detect(s["cat"], m, extra)))
    return res


def score(streams, res):
    """Per-target hits/latency, false alarms, classifier duty."""
    hits, lat, rows = [], [], []
    fa_idle = fa_events = 0
    windows = awake = 0
    for s, (m, fired) in zip(streams, res):
        windows += len(s["cat"])
        awake += len(s["cat"]) if m is None else int(m.sum())
        tg = [e for e in s["events"] if e["role"] == "target"]
        for e in tg:
            ts = [t for t, c in fired if c == e["category"] and e["onset"] <= t <= e["end"] + C.HIT_MARGIN_S]
            hits.append(bool(ts))
            lat.append(ts[0] - e["onset"] if ts else np.nan)
            rows.append(dict(stream=s["id"], env=s["env"], category=e["category"], subtype=e["subtype"], snr=e["snr"]))
        for t, c in fired:
            if not any(c == e["category"] and e["onset"] <= t <= e["end"] + C.HIT_MARGIN_S for e in tg):
                if s["kind"] == "idle":
                    fa_idle += 1
                else:
                    fa_events += 1
    return dict(hits=np.array(hits), lat=np.array(lat), rows=rows, fa_idle=fa_idle, fa_events=fa_events,
                duty=awake / max(windows, 1))


def hours(streams, kind):
    return sum(len(s["cat"]) for s in streams if s["kind"] == kind) * C.HOP / C.SR / 3600


def grid(kind):
    if kind == "F1":
        return [("F1", k) for k in C.DUTY_K]
    vals = C.FIXED_DBFS if kind == "F2" else C.NOVELTY_DB
    return [(kind, float(v), h) for v in vals for h in C.HANGOVER_S]


def calibrate(kind, tune, base_hits):
    """Lowest-duty setting whose gate-caused miss rate on tuning data is within budget."""
    best, table = None, []
    for fe in grid(kind):
        sc = score(tune, run(fe, tune))
        lost = np.sum(base_hits & ~sc["hits"]) / max(base_hits.sum(), 1)
        table.append(dict(fe=fe, duty=sc["duty"], gate_miss=float(lost), recall=float(sc["hits"].mean())))
        if lost <= C.MISS_BUDGET and (best is None or sc["duty"] < best[1]):
            best = (fe, sc["duty"])
    return (best[0] if best else None), table


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [np.nan, np.nan]
    b = rng.choice(x, (C.BOOTSTRAP, len(x))).mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mcnemar(a, b):
    """Exact McNemar test on paired hits: p-value and (only a, only b) counts."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(p=float(p), only_first=n01, only_second=n10)


def summary(fe, test, cost, rng):
    sc = score(test, run(fe, test))
    ev = [s for s in test if s["kind"] == "events"]
    h, rows = sc["hits"], sc["rows"]

    def sub(pred):
        sel = np.array([pred(r) for r in rows])
        return dict(recall=float(h[sel].mean()) if sel.any() else None, ci=boot_ci(h[sel], rng), n=int(sel.sum()))

    m_idle = [mask(fe, s) for s in test if s["kind"] == "idle"]
    idle_env = {}
    for s, m in zip([s for s in test if s["kind"] == "idle"], m_idle):
        idle_env.setdefault(s["env"], []).append(1.0 if m is None else float(m.mean()))
    m_ev = [mask(fe, s) for s in ev]
    duty_idle = float(np.mean([1.0 if m is None else m.mean() for m in m_idle]))
    duty_ev = float(np.mean([1.0 if m is None else m.mean() for m in m_ev]))
    t_yam, t_gate, t_alarm = cost
    gated = fe[0] in ("F2", "F3", "F4")
    with_alarm = fe[0] in ("F4", "F0+A")

    def compute(d):  # multiply-accumulates per window, relative to F0 (F0 = 1.0)
        return (d * t_yam + (t_gate if gated else 0) + (t_alarm if with_alarm else 0)) / t_yam

    return dict(
        frontend=list(fe), recall=float(h.mean()), recall_ci=boot_ci(h, rng),
        by_category={c: sub(lambda r, c=c: r["category"] == c) for c in C.ORDER},
        by_snr={str(v): sub(lambda r, v=v: r["snr"] == v) for v in C.SNRS_DB},
        by_cat_snr={c: {str(v): sub(lambda r, c=c, v=v: r["category"] == c and r["snr"] == v)["recall"]
                        for v in C.SNRS_DB} for c in C.ORDER},
        by_subtype={t: sub(lambda r, t=t: r["subtype"] == t) for t in sorted({r["subtype"] for r in rows})},
        by_env={e: sub(lambda r, e=e: r["env"] == e) for e in C.TEST_ENVS},
        latency_median_s=float(np.nanmedian(sc["lat"])) if np.isfinite(sc["lat"]).any() else None,
        fa_per_hour_idle=sc["fa_idle"] / hours(test, "idle"), fa_per_hour_events=sc["fa_events"] / hours(test, "events"),
        duty_idle=duty_idle, duty_events=duty_ev, duty_idle_by_env={e: float(np.mean(v)) for e, v in idle_env.items()},
        compute_idle=compute(duty_idle), compute_events=compute(duty_ev), hits=h.tolist(),
    )


def main():
    rng = np.random.default_rng(C.SEED)
    data = load()
    tune = [s for s in data if s["split"] == "tune"]
    test = [s for s in data if s["split"] == "test"]
    from classifier.yamnet import macs_per_window
    from dsp.frontend import alarm_ops_per_window, gate_ops_per_window
    cost = [macs_per_window(), gate_ops_per_window(), alarm_ops_per_window()]
    print(f"  ops per window: YAMNet {cost[0]:,}  gate {cost[1]:,}  alarm {cost[2]:,}", flush=True)
    base_tune = score(tune, run(("F0",), tune))["hits"]
    chosen, tables = {}, {}
    for kind in ("F1", "F2", "F3"):
        chosen[kind], tables[kind] = calibrate(kind, tune, base_tune)
        print(f"  {kind}: chosen {chosen[kind]}", flush=True)
    fes = {"F0": ("F0",), "F0+A": ("F0+A",), **{k: v for k, v in chosen.items() if v}}
    if chosen["F3"]:
        fes["F4"] = ("F4",) + tuple(chosen["F3"][1:])
    out = dict(ops_per_window=dict(yamnet=cost[0], gate=cost[1], alarm=cost[2]),
               hours=dict(test_events=hours(test, "events"), test_idle=hours(test, "idle"),
                          tune_events=hours(tune, "events"), tune_idle=hours(tune, "idle")),
               n_test_targets=int(sum(e["role"] == "target" for s in test for e in s["events"])),
               frontends={k: summary(fe, test, cost, rng) for k, fe in fes.items()})
    f = out["frontends"]
    h0 = np.array(f["F0"]["hits"])
    cat = np.array([r["category"] for r in score(test, run(("F0",), test))["rows"]])
    out["tests"] = {}
    for k in f:
        if k != "F0":
            hk = np.array(f[k]["hits"])
            out["tests"][f"{k} vs F0"] = dict(all=mcnemar(hk, h0),
                                              **{c: mcnemar(hk[cat == c], h0[cat == c]) for c in C.ORDER})
    # Pareto sweeps on the test set (for the figure only; the chosen points come from tuning)
    pareto = {}
    for kind in ("F1", "F2", "F3"):
        hs = [chosen[kind][2]] if kind != "F1" and chosen[kind] else C.HANGOVER_S[2:3]
        pts = []
        for fe in grid(kind):
            if kind != "F1" and fe[2] not in hs:
                continue
            sc = score(test, run(fe, test))
            pts.append(dict(fe=list(fe), duty=sc["duty"], recall=float(sc["hits"].mean())))
        pareto[kind] = pts
    out["pareto_test"] = pareto
    out["calibration"] = {k: [dict(r, fe=list(r["fe"])) for r in v] for k, v in tables.items()}
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "results.json"), "w"), indent=1)
    print_table(out)


def print_table(out):
    print(f"\nTest: {out['n_test_targets']} targets, {out['hours']['test_events']:.1f} h event streams, "
          f"{out['hours']['test_idle']:.1f} h idle streams")
    print(f"{'front end':8s} {'setting':22s} {'recall':>7s} {'fire':>6s} {'siren':>6s} {'door':>6s} "
          f"{'FA/h idle':>9s} {'FA/h ev':>8s} {'duty idle':>9s} {'duty ev':>8s} {'ops idle':>8s} {'lat':>5s}")
    for k, r in out["frontends"].items():
        bc = r["by_category"]
        print(f"{k:8s} {str(r['frontend'][1:]):22s} {r['recall']:7.3f} {bc['FIRE_ALARM']['recall']:6.3f} "
              f"{bc['SIREN']['recall']:6.3f} {bc['DOOR']['recall']:6.3f} {r['fa_per_hour_idle']:9.2f} "
              f"{r['fa_per_hour_events']:8.2f} {r['duty_idle']:9.3f} {r['duty_events']:8.3f} {r['compute_idle']:8.3f} "
              f"{r['latency_median_s'] or float('nan'):5.2f}")


if __name__ == "__main__":
    main()
