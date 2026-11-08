"""Step 2 (fast): evaluate every front end as a mask over the cached classifier outputs.

Each front end runs with one of two classifiers (fe["clf"]): "stock", YAMNet's own labels with the
app's 0.30 thresholds (or tuned ones, F0*), and "head", the trained head on YAMNet's embedding
(eval/train_head.py; thresholds chosen on the tuning streams). H* names are the head's front ends.

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
    from classifier import head, yamnet
    z = np.load(p)
    n = int(z["n"])
    bands = z["bands"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    cat = {"stock": yamnet.category_scores(z["scores"].astype(np.float32))}
    if head.available():
        cat["head"] = head.category_scores(z["emb"].astype(np.float32), z["scores"].astype(np.float32))
    s = dict(cat=cat, n=n, level=frontend.per_window(z["level"].astype(np.float32), n),
             alarms={"v4": [(float(t), "FIRE_ALARM") for t in z["alarms"]],
                     "strict": [(float(t), "FIRE_ALARM") for t in z["alarms_strict"]]},
             nov={v: frontend.gate_novelty(bands, n, d) for v, d in C.GATE_VARIANTS.items()},
             flux=frontend.onset_strength(bands), knocks={}, **meta)
    return s


def load():
    from eval.extract import DIRS
    paths = sorted(glob.glob(os.path.join(DIRS[False], "*.npz")))
    with Pool(C.WORKERS) as pool:
        return pool.map(_load, paths)


def hang(trig, h_s):
    n = int(round(h_s / (C.HOP / C.SR)))
    return np.convolve(trig.astype(int), np.ones(n + 1, int))[:len(trig)] > 0


def name(fe):
    return ("head " if fe.get("clf") == "head" else "") + _name(fe)


def _name(fe):
    k = fe["kind"]
    if k == "F1":
        return f"F1(k={fe['k']})"
    if k == "F2":
        return f"F2(T={fe['thr']:g} dBFS, H={fe['hang']} s)"
    if k in ("F3", "F4", "F5"):
        kn = fe.get("knock")
        tags = [fe["alarm"]] if fe.get("alarm") else []
        tags += ["agree"] if fe.get("agree_alarm") else []
        tags += ["siren veto"] if fe.get("veto") else []
        tags += [f"knock k={kn['kappa']} n={kn['hits']} a={kn['agree']}"] if kn else []
        return f"{k}({fe['variant']}, D={fe['thr']:g} dB, H={fe['hang']} s" + "".join(", " + t for t in tags) + ")"
    parts = ([", ".join(f"{c.lower()} {t:g}" for c, t in fe["thresholds"].items())] if fe.get("thresholds") else []) + \
        ([f"T-3 {fe['alarm']}"] if fe.get("alarm") else []) + (["siren veto"] if fe.get("veto") else []) + \
        ([f"knock k={fe['knock']['kappa']} n={fe['knock']['hits']} a={fe['knock']['agree']}"] if fe.get("knock") else [])
    return k + ("*" if fe.get("thresholds") else "") + (f"({'; '.join(parts)})" if parts else "")


def mask(fe, s):
    k = fe["kind"]
    if k == "F0":
        return None
    if k == "F1":
        return np.arange(s["n"]) % fe["k"] == 0
    if k == "F2":
        return hang(s["level"] > fe["thr"], fe["hang"])
    return hang(s["nov"][fe["variant"]] > fe["thr"], fe["hang"])


def agreed(events, cat, m, j, thr):
    """Keep DSP detections the classifier agrees with: category j scores >= thr in a classified
    window ending within AGREE_S of the detection."""
    if thr <= 0:
        return list(events)
    tw = (C.WIN + np.arange(len(cat)) * C.HOP) / C.SR
    ok = cat[:, j, 0] >= thr
    if m is not None:
        ok &= m
    return [(t, c) for t, c in events if ok[np.abs(tw - t) <= C.AGREE_S].any()]


def extras(fe, cat, m, alarms, knock_fn):
    """DSP detections merged into the alert rule for front end fe: fire alarms (optionally only
    with classifier agreement) and knocks (optionally with agreement)."""
    out = []
    if fe.get("alarm"):
        out += agreed(alarms[fe["alarm"]], cat, m, C.ORDER.index("FIRE_ALARM"), C.ALARM_AGREE if fe.get("agree_alarm") else 0)
    if fe.get("knock"):
        k = fe["knock"]
        out += agreed([(t, "DOOR") for t in knock_fn(k["kappa"], k["hits"])], cat, m, C.ORDER.index("DOOR"), k["agree"])
    return sorted(out)


def stream_knocks(s):
    def fn(kappa, hits):
        key = (kappa, hits)
        if key not in s["knocks"]:
            s["knocks"][key] = frontend.knock_times(s["flux"], kappa, hits)
        return s["knocks"][key]
    return fn


def thresholds_of(fe):
    if fe.get("thresholds"):
        return fe["thresholds"]
    if fe.get("clf") == "head":
        from classifier import head
        return head.load()["thresholds"]
    return None


def run(fe, streams):
    res = []
    thr = thresholds_of(fe)
    for s in streams:
        cat = s["cat"][fe.get("clf", "stock")]
        m = mask(fe, s)
        extra = extras(fe, cat, m, s["alarms"], stream_knocks(s))
        res.append((m, detect(cat, m, extra, thr, fe.get("veto", False))))
    return res


def score(streams, res):
    """Per-target hits/latency, false alarms (by category), classifier duty."""
    hits, lat, rows = [], [], []
    fa = {(kind, c): 0 for kind in ("idle", "events") for c in C.ORDER}
    causes = {}
    windows = awake = 0
    for s, (m, fired) in zip(streams, res):
        windows += s["n"]
        awake += s["n"] if m is None else int(m.sum())
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
    return sum(s["n"] for s in streams if s["kind"] == kind) * C.HOP / C.SR / 3600


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


def tune_thresholds(tune, clf="stock", budget=None):
    """Per category, the lowest threshold whose false alarms on the tuning streams stay within
    `budget` per hour (FA_BUDGET_PER_H). The same rule sets the head's thresholds and the tuned
    YAMNet baseline (F0*), so the two classifiers are compared at the same false-alarm budget."""
    budget = C.FA_BUDGET_PER_H if budget is None else budget
    h = hours(tune, "idle") + hours(tune, "events")
    best = {}
    for thr in C.HEAD_THRESHOLDS[::-1]:  # from strict to loose: keep the last one within budget
        fe = dict(kind="F0", clf=clf, thresholds={c: float(thr) for c in C.ORDER})
        sc = score(tune, run(fe, tune))
        for c in C.ORDER:
            fa = sum(v for (k, cc), v in sc["fa"].items() if cc == c) / h
            if fa <= budget and (c not in best or best[c][1]):
                best[c] = (float(thr), True)
            elif c in best:
                best[c] = (best[c][0], False)  # stop at the first threshold over budget
    return {c: best.get(c, (float(C.HEAD_THRESHOLDS[-1]),))[0] for c in C.ORDER}


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
    gated = fe["kind"] in ("F2", "F3", "F4", "F5")
    clf_cost = cost[0] + (cost[4] if fe.get("clf") == "head" else 0)

    def compute(duty):  # multiply-accumulates per window relative to F0
        return (duty * clf_cost + (cost[1] if gated else 0) + (cost[2] if fe.get("alarm") else 0)
                + (cost[3] if fe.get("knock") else 0)) / cost[0]


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


def knock_search(tune, base_fe):
    """Knock detector setting: most door recall on tuning streams with no more false alarms than
    the system without it (KNOCK_GRID)."""
    b = score(tune, run(base_fe, tune))
    fa0 = sum(b["fa"].values())
    cats = np.array([r["category"] for r in b["rows"]])
    table, best = {}, None
    for kappa in C.KNOCK_GRID["kappa"]:
        for hits in C.KNOCK_GRID["hits"]:
            for agree in C.KNOCK_GRID["agree"]:
                kfe = dict(base_fe, knock=dict(kappa=kappa, hits=hits, agree=agree))
                sc = score(tune, run(kfe, tune))
                fa = sum(sc["fa"].values())
                door = float(sc["hits"][cats == "DOOR"].mean())
                table[f"{kappa}/{hits}/{agree}"] = dict(door=door, fa=fa)
                if fa <= fa0 and (best is None or (door, -fa, kappa) > (best[0], -best[1], best[2]["kappa"])):
                    best = (door, fa, kfe["knock"])
    base_door = float(b["hits"][cats == "DOOR"].mean())
    print(f"  knock detector ({base_fe.get('clf', 'stock')}): tuning door recall {base_door:.3f} -> "
          f"{best[0] if best else float('nan'):.3f} (false alarms {fa0} -> {best[1] if best else '-'}), chosen {best and best[2]}",
          flush=True)
    return dict(tune_base_door=base_door, tune_base_fa=fa0, chosen=best and best[2], tune_table=table)


def misses(fe_hits, ref_hits, rows):
    """Why the proposed system misses a target, per category: the gate slept (the same classifier
    always on catches it) or the classifier itself misses it."""
    out = {}
    for c in C.ORDER:
        sel = np.array([r["category"] == c for r in rows])
        miss = sel & ~fe_hits
        out[c] = dict(n=int(sel.sum()), missed=int(miss.sum()), gate_slept=int((miss & ref_hits).sum()),
                      classifier=int((miss & ~ref_hits).sum()))
    return out


def main():
    from classifier import head
    rng = np.random.default_rng(C.SEED)
    data = load()
    tune = [s for s in data if s["split"] == "tune"]
    test = [s for s in data if s["split"] == "test"]
    clfs = ["stock"] + (["head"] if head.available() else [])
    from classifier.yamnet import macs_per_window
    cost = [macs_per_window(), frontend.gate_ops_per_window(), frontend.alarm_ops_per_window(frontend.chosen_alarm()),
            frontend.knock_ops_per_window(), head.macs() if head.available() else 0]
    print(f"  ops per window: YAMNet {cost[0]:,}  gate {cost[1]:,}  alarm {cost[2]:,}  knock {cost[3]:,}  head {cost[4]:,}",
          flush=True)

    # --- calibration on tuning data -------------------------------------------------------------
    base = {c: score(tune, run(dict(kind="F0", clf=c), tune))["hits"] for c in clfs}
    print(f"  tuning: {len(base['stock'])} targets, always-on detects " +
          ", ".join(f"{c} {int(b.sum())}" for c, b in base.items()), flush=True)
    tables = {"F1": sweep(grid("F1"), tune, base["stock"]), "F2": sweep(grid("F2"), tune, base["stock"])}
    for v in C.GATE_VARIANTS:
        tables[f"F3:{v}"] = sweep(grid("F3", v), tune, base["stock"])
    print("\n  Tuning environments, lowest classifier duty within each gate-caused miss budget:")
    print(f"  {'gate':22s}" + "".join(f"{f'{100 * b:g} %':>10s}" for b in C.BUDGETS))
    for k, t in tables.items():
        print(f"  {k:22s}" + "".join(f"{(100 * c['duty']) if (c := choose(t, b)) else float('nan'):9.1f}%" for b in C.BUDGETS))
    variant = min(C.GATE_VARIANTS, key=lambda v: (choose(tables[f"F3:{v}"], C.VARIANT_BUDGET) or {"duty": 9})["duty"])
    print(f"  -> selected gate variant: {variant}\n", flush=True)
    if "head" in clfs:  # the same gate statistic, recalibrated against the head's own detections
        tables["H3"] = sweep([dict(fe, clf="head") for fe in grid("F3", variant)], tune, base["head"])

    # --- front ends to report on the test set ---------------------------------------------------
    thresholds = tune_thresholds(tune)
    print(f"  tuned YAMNet thresholds ({C.FA_BUDGET_PER_H} FA/h per category on tuning): {thresholds}", flush=True)
    if "head" in clfs:
        print(f"  head thresholds: {head.load()['thresholds']}", flush=True)
    fes = {"F0": dict(kind="F0"), "F0*": dict(kind="F0", thresholds=thresholds),
           "F0+DSP": dict(kind="F0", alarm="v4", veto=True)}
    for b in C.BUDGETS:
        tag = f"@{100 * b:g}%"
        for k, tk in (("F1", "F1"), ("F2", "F2"), ("F3", f"F3:{variant}")):
            c = choose(tables[tk], b)
            if c:
                fes[k + tag] = c["fe"]
        if "head" in clfs and (c := choose(tables["H3"], b)):
            fes["H3" + tag] = c["fe"]
    knock = {}
    for clf, g, a in (("stock", "F3@5%", "F4"), ("head", "H3@5%", "H4")):
        if g not in fes:
            continue
        fes[f"{a}@5%"] = dict(fes[g], kind="F4", alarm="v4", veto=True)
        knock[clf] = knock_search(tune, fes[f"{a}@5%"])
        if knock[clf]["chosen"]:
            fes[f"{a} knock@5%"] = dict(fes[f"{a}@5%"], knock=knock[clf]["chosen"])
    if "head" in clfs:
        fes["H0"] = dict(kind="F0", clf="head")
        fes["H0+DSP"] = dict(kind="F0", clf="head", alarm="v4", veto=True, **({"knock": knock["head"]["chosen"]}
                                                                              if knock.get("head", {}).get("chosen") else {}))
    if knock.get("stock", {}).get("chosen"):
        fes["F0+DSP"]["knock"] = knock["stock"]["chosen"]
    out = dict(ops_per_window=dict(yamnet=cost[0], gate=cost[1], alarm=cost[2], knock=cost[3], head=cost[4]),
               tuned_thresholds=thresholds, head_thresholds=head.load()["thresholds"] if "head" in clfs else None,
               hours={f"{sp}_{k}": hours(ss, k) for sp, ss in (("test", test), ("tune", tune)) for k in ("events", "idle")},
               n_test_targets=int(sum(e["role"] == "target" for s in test for e in s["events"])),
               selected_variant=variant, frontends={k: summary(fe, test, cost, rng) for k, fe in fes.items()})
    f = out["frontends"]
    rows = score(test, run(dict(kind="F0"), test))["rows"]
    cats = np.array([r["category"] for r in rows])

    def test_pair(a, b):
        ha, hb = np.array(f[a]["hits"]), np.array(f[b]["hits"])
        return dict(all=mcnemar(ha, hb), **{c: mcnemar(ha[cats == c], hb[cats == c]) for c in C.ORDER})
    out["tests"] = {f"{k} vs F0": test_pair(k, "F0") for k in f if k != "F0"}
    for a, b in (("H0", "F0*"), ("H4 knock@5%", "H0"), ("H4 knock@5%", "F4 knock@5%"), ("H0+DSP", "H0"), ("F0+DSP", "F0")):
        if a in f and b in f:
            out["tests"][f"{a} vs {b}"] = test_pair(a, b)
    out["misses"] = {k: misses(np.array(f[k]["hits"], bool), np.array(f[ref]["hits"], bool), rows)
                     for k, ref in (("F4 knock@5%", "F0+DSP"), ("H4 knock@5%", "H0+DSP")) if k in f and ref in f}
    # Ablation of gate variants at the 1 % budget (setting chosen on tuning, measured on test)
    out["ablation"] = {}
    for v in C.GATE_VARIANTS:
        c = choose(tables[f"F3:{v}"], C.BUDGETS[0])
        if c:
            s = summary(c["fe"], test, cost, rng)
            out["ablation"][v] = dict(setting=name(c["fe"]), tune_duty=c["duty"], tune_gate_miss=c["gate_miss"],
                                      test_duty_idle=s["duty_idle"], test_duty_events=s["duty_events"],
                                      test_recall=s["recall"], test_duty_idle_by_env=s["duty_idle_by_env"])
    # Trade-off curves on the test set (figure only; chosen points come from tuning)
    h0 = np.array(f["F0"]["hits"], bool)
    out["pareto_test"] = {}
    for key, kind, v, clf in (("F1", "F1", None, "stock"), ("F2", "F2", None, "stock"), ("F3 v1", "F3", "v1", "stock"),
                              ("F3", "F3", variant, "stock"), ("H3", "F3", variant, "head")):
        if clf not in clfs:
            continue
        cc = choose(tables["F2" if kind == "F2" else ("H3" if clf == "head" else f"F3:{v}") if kind == "F3" else "F1"],
                    C.BUDGETS[0])
        hs = cc["fe"].get("hang") if cc else None
        fes_k = [dict(fe, clf=clf) for fe in grid(kind, v) if hs is None or fe.get("hang") == hs]
        ref = h0 if clf == "stock" else np.array(f["H0"]["hits"], bool)
        out["pareto_test"][key] = [dict(fe=r["fe"], duty=r["duty"], recall=r["recall"]) for r in sweep(fes_k, test, ref)]
    out["calibration"] = tables
    out["knock"] = knock
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "results.json"), "w"), indent=1)
    print_table(out)


def print_table(out):
    print(f"Test: {out['n_test_targets']} targets, {out['hours']['test_events']:.1f} h event streams, "
          f"{out['hours']['test_idle']:.1f} h idle streams")
    short = {"FIRE_ALARM": "alarm", "SIREN": "siren", "CAR_HORN": "horn", "BABY_CRY": "baby", "DOOR": "door"}
    print(f"{'front end':14s} {'recall':>7s} {'95% CI':>13s} " + "".join(f"{short[c]:>6s} " for c in C.ORDER) +
          f"{'FA/h id':>7s} {'FA/h ev':>7s} {'duty id':>7s} {'duty ev':>7s} {'ops id':>6s} {'lat':>5s}  setting")
    for k, r in out["frontends"].items():
        bc = r["by_category"]
        ci = r["recall_ci"]
        print(f"{k:14s} {r['recall']:7.3f} [{ci[0]:.3f},{ci[1]:.3f}] " +
              "".join(f"{bc[c]['recall']:6.3f} " for c in C.ORDER) +
              f"{r['fa_per_hour_idle']:7.2f} "
              f"{r['fa_per_hour_events']:7.2f} {r['duty_idle']:7.3f} {r['duty_events']:7.3f} {r['compute_idle']:6.3f} "
              f"{r['latency_median_s'] or float('nan'):5.2f}  {r['name']}")
    for k, m in out.get("misses", {}).items():
        print(f"  misses of {k}: " + ", ".join(f"{c.lower()} {v['missed']}/{v['n']} (gate {v['gate_slept']})"
                                              for c, v in m.items()))
    print("\nGate variants at the 1 % budget (chosen on tuning, measured on test):")
    for v, a in out["ablation"].items():
        print(f"  {v:14s} tune duty {a['tune_duty']:.3f}  test duty idle {a['test_duty_idle']:.3f} "
              f"events {a['test_duty_events']:.3f}  test recall {a['test_recall']:.3f}  {a['setting']}")


if __name__ == "__main__":
    main()
