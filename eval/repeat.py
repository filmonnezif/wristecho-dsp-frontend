"""Check (not in the paper yet): does recall rise when a sound repeats, and what is the clean ceiling?

For every test target (same clip or alarm seed, room, microphone, level and SNR as in its test
stream): the sound is repeated with 2 s gaps for 60 s after a 5 s lead-in of the room, and the
final system (best classifier per category at 0.3 FA/h + two-floor gate at the 5 % budget + DSP
alarm and knock detectors + siren veto, settings from the tuning streams) runs on it. Reported:
share caught within 8, 30 and 60 s of the first onset. Clean ceiling: the sound once, alone
(room at -60 dB below it), with stock YAMNet (app rule) and the best classifier always on.
Writes results/repeat.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import head, yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402
from eval import evaluate as E  # noqa: E402
from eval import improve  # noqa: E402

LEAD, LENGTH, GAP = 5.0, 60.0, 2.0
SETTINGS = {}


def cats(x):
    scores, emb = yamnet.window_outputs(x)
    s = {"cat": {"stock": yamnet.category_scores(scores), "head": head.category_scores(emb, scores)}}
    improve.add_variants(s)
    best = np.stack([s["cat"][SETTINGS["choice"][c]][:, j] for j, c in enumerate(C.ORDER)], axis=1)
    return s["cat"]["stock"], best


def first_hit(fired, cat):
    ts = [t for t, c in fired if c == cat and t >= LEAD]
    return ts[0] - LEAD if ts else None


def job(arg):
    spec, ev = arg
    fe, thr = SETTINGS["fe"], SETTINGS["thr"]
    a = streams.event_audio(ev["source"])
    bg = streams.load(os.path.join(C.DATA, "demand", spec["env"], f"ch{spec['ch']:02d}.wav"))
    bg = bg / (np.sqrt(np.mean(bg ** 2)) + 1e-9) * 10 ** (spec["bg_dbfs"] / 20)
    n = int((LEAD + LENGTH) * C.SR)
    x = bg[:n].copy()
    g = 10 ** (spec["bg_dbfs"] / 20) * 10 ** (ev["snr"] / 20)
    t = LEAD
    while t < LEAD + LENGTH:
        i = int(t * C.SR)
        seg = a[:n - i]
        x[i:i + len(seg)] += seg * g
        t += len(a) / C.SR + GAP
    x = np.clip(x, -1, 1).astype(np.float32)
    _, best = cats(x)
    _, bands = frontend.band_power(x)
    nw = yamnet.n_windows(len(x))
    stat = frontend.gate_novelty(bands, nw, C.GATE_VARIANTS[fe["variant"]])
    k = int(round(fe["hang"] / (C.HOP / C.SR)))
    m = np.convolve((stat > fe["thr"]).astype(int), np.ones(k + 1, int))[:nw] > 0
    cfg = frontend.chosen_alarm()
    alarms = {"v4": [(u, "FIRE_ALARM") for u in frontend.decide(frontend.tone_track(x, cfg), cfg["rule"])]}
    flux = frontend.onset_strength(bands)
    ex = E.extras(fe, best, m, alarms, lambda kappa, hits: frontend.knock_times(flux, kappa, hits))
    lat = first_hit(yamnet.detect(best, m, ex, thr, siren_veto=True), ev["category"])
    # clean ceiling: the sound once, alone
    y = np.r_[np.zeros(int(1.0 * C.SR)), a, np.zeros(int(1.0 * C.SR))].astype(np.float32) * 0.1
    y += 1e-3 * 0.1 * np.random.default_rng(0).standard_normal(len(y)).astype(np.float32)
    st_c, best_c = cats(y)
    clean_stock = any(c == ev["category"] for _, c in yamnet.detect(st_c, siren_veto=True))
    clean_best = any(c == ev["category"] for _, c in yamnet.detect(best_c, None, (), thr, siren_veto=True))
    return dict(category=ev["category"], snr=ev["snr"], latency=lat, clean_stock=clean_stock, clean_best=clean_best)


def main():
    imp = json.load(open(os.path.join(C.RESULTS, "improve.json")))
    choice, thr = imp["choice"], imp["curve"]["best"]["0.3"]["thresholds"]
    SETTINGS.update(choice=choice, thr=thr)
    improve.DATA = E.load()
    for s in improve.DATA:
        improve.add_variants(s)
        s["cat"]["best"] = np.stack([s["cat"][choice[c]][:, j] for j, c in enumerate(C.ORDER)], axis=1)
    tune = improve.tune_split()
    on = dict(kind="F0", clf="best", veto=True, thresholds=thr)
    base = E.score(tune, E.run(on, tune))["hits"]
    table = E.sweep([dict(fe, clf="best", thresholds=thr) for fe in E.grid("F3", "two-floor+6")], tune, base)
    fe = dict(E.choose(table, 0.05)["fe"], kind="F4", alarm="v4", veto=True)
    kn = E.knock_search(tune, fe)["chosen"]
    if kn:
        fe["knock"] = kn
    SETTINGS["fe"] = fe
    improve.DATA = []
    jobs = [(sp, ev) for sp in streams.specs() if sp["split"] == "test" and sp["kind"] == "events"
            for ev in sp["events"] if ev["role"] == "target"]
    with Pool(C.WORKERS) as pool:
        res = pool.map(job, jobs)
    json.dump(dict(settings=dict(fe=fe, thresholds=thr, choice=choice), events=res),
              open(os.path.join(C.RESULTS, "repeat.json"), "w"), indent=1)

    def row(name, rs):
        lat = [r["latency"] for r in rs]
        within = lambda s: np.mean([v is not None and v <= s for v in lat])  # noqa: E731
        print(f"  {name:12s} n={len(rs):3d}  caught within 8 s {within(8):.2f}  30 s {within(30):.2f}  60 s {within(60):.2f}"
              f"  | clean, once: stock {np.mean([r['clean_stock'] for r in rs]):.2f}  best {np.mean([r['clean_best'] for r in rs]):.2f}")
    row("all", res)
    for c in C.ORDER:
        row(c.lower(), [r for r in res if r["category"] == c])
    for snr in C.SNRS_DB:
        row(f"{snr} dB", [r for r in res if r["snr"] == snr])


if __name__ == "__main__":
    main()
