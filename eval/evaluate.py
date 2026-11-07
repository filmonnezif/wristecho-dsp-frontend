"""Step 2 (fast): evaluate every front end as a mask over the cached classifier scores.

Front ends
  F0      classifier on every window (the deployed baseline)
  F1      classifier on every k-th window (naive duty cycling)
  F2      fixed energy gate: wake when the HPF'd level exceeds T dBFS
  F3      adaptive gate: wake when the novelty statistic exceeds Delta (variant chosen on tuning data)
  F4      F3 + the Goertzel alarm detector (redesigned on real recordings, eval/alarm_design.py) +
          the siren veto (proposed); "v3" = the first detector (T-3 strict, 2.8-3.6 kHz)
  F0+A    classifier on every window + the alarm detector (ablation: separates the two effects)
  F0*     classifier on every window with thresholds tuned on the tuning streams (stronger baseline)
Gates keep the classifier awake for a hangover H after the last trigger.

Calibration uses the tuning environments only. For a budget b, a gate's setting is the one with
the lowest classifier duty such that at most b of the events F0 detects are lost because the gate
slept. The gate variant is the one with the lowest tuning duty at the 1 % budget. Everything
reported is measured on the test environments, never seen while tuning.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier.yamnet import detect  # noqa: E402
from dsp import frontend  # noqa: E402


def _load(p):
    from eval.extract import alarm_dir
    z = np.load(p)
    a4 = os.path.join(alarm_dir(frontend.chosen_alarm()), os.path.basename(p)[:-4] + ".npy")
    n = int(z["n"])
    bands = z["bands"].astype(np.float32)
    s = dict(cat=z["cat"].astype(np.float32), level=frontend.per_window(z["level"].astype(np.float32), n),
             alarms={"loose": [(float(t), "FIRE_ALARM") for t in z["alarms"]],
                     "strict": [(float(t), "FIRE_ALARM") for t in z["alarms_strict"]],
                     "v4": [(float(t), "FIRE_ALARM") for t in (np.load(a4) if os.path.exists(a4) else z["alarms_strict"])]},
             nov={v: frontend.gate_novelty(bands, n, d) for v, d in C.GATE_VARIANTS.items()},
             **json.loads(str(z["meta"])))
    return s


def load():
    paths = sorted(glob.glob(os.path.join(C.CACHE, "v2", "*.npz")))
    with Pool(C.WORKERS) as pool:
        return pool.map(_load, paths)


def hang(trig, h_s):
    n = int(round(h_s / (C.HOP / C.SR)))
    return np.convolve(trig.astype(int), np.ones(n + 1, int))[:len(trig)] > 0


def name(fe):
    k = fe["kind"]
    if k == "F1":
        return f"F1(k={fe['k']})"
    if k == "F2":
        return f"F2(T={fe['thr']:g} dBFS, H={fe['hang']} s)"
    if k in ("F3", "F4"):
        return (f"{k}({fe['variant']}, D={fe['thr']:g} dB, H={fe['hang']} s{', ' + fe['alarm'] if fe.get('alarm') else ''}"
                f"{', siren veto' if fe.get('veto') else ''})")
    parts = ([", ".join(f"{c.lower()} {t:g}" for c, t in fe["thresholds"].items())] if fe.get("thresholds") else []) + \
        ([f"T-3 {fe['alarm']}"] if fe.get("alarm") else []) + (["siren veto"] if fe.get("veto") else [])
    return k + ("*" if fe.get("thresholds") else "") + (f"({'; '.join(parts)})" if parts else "")


def mask(fe, s):
    k = fe["kind"]
    if k == "F0":
        return None
    if k == "F1":
        return np.arange(len(s["cat"])) % fe["k"] == 0
    if k == "F2":
        return hang(s["level"] > fe["thr"], fe["hang"])
    return hang(s["nov"][fe["variant"]] > fe["thr"], fe["hang"])


def run(fe, streams):
    res = []
    for s in streams:
        m = mask(fe, s)
        extra = s["alarms"][fe["alarm"]] if fe.get("alarm") else ()
        res.append((m, detect(s["cat"], m, extra, fe.get("thresholds"), fe.get("veto", False))))
    return res


def score(streams, res):
    """Per-target hits/latency, false alarms (by category), classifier duty."""
    hits, lat, rows = [], [], []
    fa = {(kind, c): 0 for kind in ("idle", "events") for c in C.ORDER}
    causes = {}
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
                fa[(s["kind"], c)] += 1
                # cause: the sound playing at that moment (another target, a distractor, or background)
                during = [e for e in s["events"] if e["onset"] <= t <= e["end"] + C.HIT_MARGIN_S]
                e = next((e for e in during if e["role"] == "target"), during[0] if during else None)
                cause = "background" if e is None else (
                    f"{e['category'].lower()} target" if e["role"] == "target" else f"{e['subtype']} distractor")
                causes[f"{c} during {cause}"] = causes.get(f"{c} during {cause}", 0) + 1
    return dict(hits=np.array(hits), lat=np.array(lat), rows=rows, fa=fa, causes=causes, duty=awake / max(windows, 1))


def hours(streams, kind):
    return sum(len(s["cat"]) for s in streams if s["kind"] == kind) * C.HOP / C.SR / 3600


def grid(kind, variant=None):
    if kind == "F1":
        return [dict(kind="F1", k=k) for k in C.DUTY_K]
    if kind == "F2":
        return [dict(kind="F2", thr=float(t), hang=h) for t in C.FIXED_DBFS for h in C.HANGOVER_S]
    return [dict(kind="F3", variant=variant, thr=float(t), hang=h) for t in C.NOVELTY_DB for h in C.HANGOVER_S]


def sweep(fes, streams, base_hits):
    out = []
    for fe in fes:
        sc = score(streams, run(fe, streams))
        lost = float(np.sum(base_hits & ~sc["hits"]) / max(base_hits.sum(), 1))
        out.append(dict(fe=fe, duty=sc["duty"], gate_miss=lost, recall=float(sc["hits"].mean())))
    return out


def choose(table, budget):
    ok = [r for r in table if r["gate_miss"] <= budget + 1e-12]
    return min(ok, key=lambda r: (r["duty"], -r["recall"])) if ok else None


def tune_thresholds(tune):
    """Tuned-baseline ablation: per category, the lowest YAMNet threshold whose false alarms on the
    tuning streams do not exceed those of the app's 0.30 (so recall rises at no extra false alarms)."""
    rows = []
    for thr in C.THRESHOLD_GRID:
        fe = dict(kind="F0", thresholds={c: float(thr) for c in C.ORDER})
        sc = score(tune, run(fe, tune))
        cats = np.array([r["category"] for r in sc["rows"]])
        fa = {c: sum(v for (k, cc), v in sc["fa"].items() if cc == c) for c in C.ORDER}
        rows.append((float(thr), {c: float(sc["hits"][cats == c].mean()) for c in C.ORDER}, fa))
    ref = next(r for r in rows if abs(r[0] - 0.30) < 1e-9)
    return {c: min(t for t, rec, fa in rows if fa[c] <= ref[2][c]) for c in C.ORDER}


def cluster_ci(h, groups, rng):
    """95 % bootstrap interval resampling whole streams (events in one stream share a background)."""
    h = np.asarray(h, float)
    if len(h) == 0:
        return [None, None]
    keys = sorted(set(groups))
    idx = {k: i for i, k in enumerate(keys)}
    g = np.array([idx[x] for x in groups])
    sums, cnt = np.bincount(g, h, len(keys)), np.bincount(g, None, len(keys))
    b = rng.integers(0, len(keys), (C.BOOTSTRAP, len(keys)))
    m = sums[b].sum(1) / np.maximum(cnt[b].sum(1), 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar(a, b):
    """Exact McNemar test on paired per-event hits."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(p=float(p), only_first=n01, only_second=n10)


def summary(fe, test, cost, rng):
    sc = score(test, run(fe, test))
    h, rows = sc["hits"], sc["rows"]
    grp = [r["stream"] for r in rows]

    def sub(pred):
        sel = np.array([pred(r) for r in rows])
        return dict(recall=float(h[sel].mean()) if sel.any() else None,
                    ci=cluster_ci(h[sel], [g for g, s in zip(grp, sel) if s], rng), n=int(sel.sum()))

    idle = [s for s in test if s["kind"] == "idle"]
    ev = [s for s in test if s["kind"] == "events"]
    d = lambda ss: [1.0 if (m := mask(fe, s)) is None else float(m.mean()) for s in ss]  # noqa: E731
    d_idle, d_ev = d(idle), d(ev)
    by_env = {}
    for s, v in zip(idle, d_idle):
        by_env.setdefault(s["env"], []).append(v)
    gated = fe["kind"] in ("F2", "F3", "F4")

    def compute(duty):  # multiply-accumulates per window relative to F0
        return (duty * cost[0] + (cost[1] if gated else 0) + (cost[2] if fe.get("alarm") else 0)) / cost[0]

    h_idle, h_ev = hours(test, "idle"), hours(test, "events")
    return dict(
        name=name(fe), frontend=fe, recall=float(h.mean()), recall_ci=cluster_ci(h, grp, rng),
        recall_snr_ge5=sub(lambda r: r["snr"] >= 5),
        by_category={c: sub(lambda r, c=c: r["category"] == c) for c in C.ORDER},
        by_snr={str(v): sub(lambda r, v=v: r["snr"] == v) for v in C.SNRS_DB},
        by_cat_snr={c: {str(v): sub(lambda r, c=c, v=v: r["category"] == c and r["snr"] == v)["recall"]
                        for v in C.SNRS_DB} for c in C.ORDER},
        by_subtype={t: sub(lambda r, t=t: r["subtype"] == t) for t in sorted({r["subtype"] for r in rows})},
        by_env={e: sub(lambda r, e=e: r["env"] == e) for e in C.TEST_ENVS},
        latency_median_s=float(np.nanmedian(sc["lat"])) if np.isfinite(sc["lat"]).any() else None,
        fa_per_hour_idle=sum(v for (k, _), v in sc["fa"].items() if k == "idle") / h_idle,
        fa_per_hour_events=sum(v for (k, _), v in sc["fa"].items() if k == "events") / h_ev,
        fa_per_hour_by_category={f"{k}:{c}": v / (h_idle if k == "idle" else h_ev) for (k, c), v in sc["fa"].items()},
        fa_count_by_cause=dict(sorted(sc["causes"].items(), key=lambda kv: -kv[1])),
        duty_idle=float(np.mean(d_idle)), duty_events=float(np.mean(d_ev)),
        duty_idle_by_env={e: float(np.mean(v)) for e, v in by_env.items()},
        compute_idle=compute(float(np.mean(d_idle))), compute_events=compute(float(np.mean(d_ev))),
        hits=h.tolist(),
    )


def main():
    rng = np.random.default_rng(C.SEED)
    data = load()
    tune = [s for s in data if s["split"] == "tune"]
    test = [s for s in data if s["split"] == "test"]
    from classifier.yamnet import macs_per_window
    cost = [macs_per_window(), frontend.gate_ops_per_window(), frontend.alarm_ops_per_window(frontend.chosen_alarm())]
    print(f"  ops per window: YAMNet {cost[0]:,}  gate {cost[1]:,}  alarm {cost[2]:,}", flush=True)
    base = score(tune, run(dict(kind="F0"), tune))["hits"]
    print(f"  tuning: {len(base)} targets, F0 detects {int(base.sum())}", flush=True)

    # --- calibration on tuning data -------------------------------------------------------------
    tables = {"F1": sweep(grid("F1"), tune, base), "F2": sweep(grid("F2"), tune, base)}
    for v in C.GATE_VARIANTS:
        tables[f"F3:{v}"] = sweep(grid("F3", v), tune, base)
    print("\n  Tuning environments, lowest classifier duty within each gate-caused miss budget:")
    print(f"  {'gate':22s}" + "".join(f"{f'{100 * b:g} %':>10s}" for b in C.BUDGETS))
    for k, t in tables.items():
        print(f"  {k:22s}" + "".join(f"{(100 * c['duty']) if (c := choose(t, b)) else float('nan'):9.1f}%" for b in C.BUDGETS))
    variant = min(C.GATE_VARIANTS, key=lambda v: (choose(tables[f"F3:{v}"], C.BUDGETS[0]) or {"duty": 9})["duty"])
    print(f"  -> selected gate variant: {variant}\n", flush=True)

    # --- front ends to report on the test set ---------------------------------------------------
    thresholds = tune_thresholds(tune)
    print(f"  tuned YAMNet thresholds (no more tuning false alarms than 0.30): {thresholds}", flush=True)
    fes = {"F0": dict(kind="F0"), "F0 veto": dict(kind="F0", veto=True),
           "F0+A": dict(kind="F0", alarm="v4", veto=True), "F0+A v3": dict(kind="F0", alarm="strict", veto=True),
           "F0*": dict(kind="F0", thresholds=thresholds)}
    for b in C.BUDGETS:
        tag = f"@{100 * b:g}%"
        for k, tk in (("F1", "F1"), ("F2", "F2"), ("F3", f"F3:{variant}")):
            c = choose(tables[tk], b)
            if c:
                fes[k + tag] = c["fe"]
        if f"F3{tag}" in fes:
            fes[f"F4{tag}"] = dict(fes[f"F3{tag}"], kind="F4", alarm="v4", veto=True)
    c = choose(tables["F3:v1"], C.BUDGETS[0])
    if c:
        fes["F3 v1@1%"] = c["fe"]
    if "F3@5%" in fes:
        fes["F4 v3@5%"] = dict(fes["F3@5%"], kind="F4", alarm="strict", veto=True)
    out = dict(ops_per_window=dict(yamnet=cost[0], gate=cost[1], alarm=cost[2]), tuned_thresholds=thresholds,
               hours={f"{sp}_{k}": hours(ss, k) for sp, ss in (("test", test), ("tune", tune)) for k in ("events", "idle")},
               n_test_targets=int(sum(e["role"] == "target" for s in test for e in s["events"])),
               selected_variant=variant,
               frontends={k: summary(fe, test, cost, rng) for k, fe in fes.items()})
    f = out["frontends"]
    h0 = np.array(f["F0"]["hits"])
    cats = np.array([r["category"] for r in score(test, run(dict(kind="F0"), test))["rows"]])
    out["tests"] = {f"{k} vs F0": dict(all=mcnemar(np.array(v["hits"]), h0),
                                        **{c: mcnemar(np.array(v["hits"])[cats == c], h0[cats == c]) for c in C.ORDER})
                    for k, v in f.items() if k != "F0"}
    # Ablation of gate variants at the 1 % budget (setting chosen on tuning, measured on test)
    out["ablation"] = {}
    for v in C.GATE_VARIANTS:
        c = choose(tables[f"F3:{v}"], C.BUDGETS[0])
        if c:
            s = summary(c["fe"], test, cost, rng)
            out["ablation"][v] = dict(setting=name(c["fe"]), tune_duty=c["duty"], tune_gate_miss=c["gate_miss"],
                                      test_duty_idle=s["duty_idle"], test_duty_events=s["duty_events"],
                                      test_recall=s["recall"], test_duty_idle_by_env=s["duty_idle_by_env"])
    c2 = choose(tables["F2"], C.BUDGETS[0])
    # Trade-off curves on the test set (figure only; chosen points come from tuning)
    out["pareto_test"] = {}
    for key, kind, v in (("F1", "F1", None), ("F2", "F2", None), ("F3 v1", "F3", "v1"), ("F3", "F3", variant)):
        hs = None
        if kind == "F2" and c2:
            hs = c2["fe"]["hang"]
        elif kind == "F3":
            cc = choose(tables[f"F3:{v}"], C.BUDGETS[0])
            hs = cc["fe"]["hang"] if cc else None
        fes_k = [fe for fe in grid(kind, v) if hs is None or fe.get("hang") == hs]
        out["pareto_test"][key] = [dict(fe=r["fe"], duty=r["duty"], recall=r["recall"])
                                   for r in sweep(fes_k, test, h0)]
    out["calibration"] = tables
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "results.json"), "w"), indent=1)
    print_table(out)


def print_table(out):
    print(f"Test: {out['n_test_targets']} targets, {out['hours']['test_events']:.1f} h event streams, "
          f"{out['hours']['test_idle']:.1f} h idle streams")
    print(f"{'front end':14s} {'recall':>7s} {'95% CI':>13s} {'fire':>6s} {'siren':>6s} {'door':>6s} "
          f"{'FA/h id':>7s} {'FA/h ev':>7s} {'duty id':>7s} {'duty ev':>7s} {'ops id':>6s} {'lat':>5s}  setting")
    for k, r in out["frontends"].items():
        bc = r["by_category"]
        ci = r["recall_ci"]
        print(f"{k:14s} {r['recall']:7.3f} [{ci[0]:.3f},{ci[1]:.3f}] {bc['FIRE_ALARM']['recall']:6.3f} "
              f"{bc['SIREN']['recall']:6.3f} {bc['DOOR']['recall']:6.3f} {r['fa_per_hour_idle']:7.2f} "
              f"{r['fa_per_hour_events']:7.2f} {r['duty_idle']:7.3f} {r['duty_events']:7.3f} {r['compute_idle']:6.3f} "
              f"{r['latency_median_s'] or float('nan'):5.2f}  {r['name']}")
    print("\nGate variants at the 1 % budget (chosen on tuning, measured on test):")
    for v, a in out["ablation"].items():
        print(f"  {v:14s} tune duty {a['tune_duty']:.3f}  test duty idle {a['test_duty_idle']:.3f} "
              f"events {a['test_duty_events']:.3f}  test recall {a['test_recall']:.3f}  {a['setting']}")


if __name__ == "__main__":
    main()
