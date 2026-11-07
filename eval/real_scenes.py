"""Step 2d: the proposed system on real, naturally recorded audio (nothing mixed in).

DESED public evaluation set: 692 real 10 s home recordings (YouTube, Creative Commons) with
  time-stamped labels of 10 classes. Per labelled event: was the gate awake while it sounded
  ("wake recall")? Per clip: classifier duty, and how many of the always-on classifier's alerts
  the gated system keeps. Each clip gets a 2 s warm-up (excluded) for the noise floor.
SINS (DCASE 2018 Task 5): recordings of one real home, 10 s segments labelled with the household's
  activity (absence, watching TV, cooking, ...); up to 400 segments per activity from the first
  audio part (which holds scattered segments, so each is scored on its own after a 2 s
  warm-up). Per activity: classifier duty and alerts per hour of real home life, for the
  always-on classifier and the proposed system.
The gate and alarm detector settings are the ones chosen earlier (results.json, F4 at the 5 %
budget; results/alarm_design.json); nothing is tuned on this data.
"""
import csv
import glob
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402

DESED_URL = "https://zenodo.org/api/records/4560759/files/DESED_public_eval.tar.gz/content"
SINS_URL = "https://zenodo.org/api/records/1247102/files/DCASE2018-task5-dev.{part}.zip/content"
SINS_PARTS = ["meta", "audio.1"]  # 1 of 23 audio parts (2 GB, ~3,200 segments; Zenodo is slow)
PER_ACTIVITY = 400  # SINS segments per activity (10 s each)
WARMUP_S = 2.0


def proposed_gate(budget="5%"):
    """The proposed system's settings from results.json (F5 if the knock detector was chosen)."""
    p = os.path.join(C.RESULTS, "results.json")
    if not os.path.exists(p):
        return None
    f = json.load(open(p))["frontends"]
    return (f.get(f"F5@{budget}") or f[f"F4@{budget}"])["frontend"]


def gate_mask(x, fe, bands=None):
    """Classifier windows the gate keeps awake for audio x, with front end fe (kind F3/F4/F5)."""
    n = yamnet.n_windows(len(x))
    if bands is None:
        _, bands = frontend.band_power(x)
    stat = frontend.gate_novelty(bands, n, C.GATE_VARIANTS[fe["variant"]])
    trig = stat > fe["thr"]
    k = int(round(fe["hang"] / (C.HOP / C.SR)))
    return np.convolve(trig.astype(int), np.ones(k + 1, int))[:n] > 0


def system(x, fe, cfg, cat=None):
    """Always-on classifier (F0, siren veto) and the proposed system on audio x: gate, DSP alarm
    detector (with classifier agreement if fe says so), knock detector, siren veto."""
    from eval.evaluate import extras
    if cat is None:
        cat = yamnet.category_scores(yamnet.window_scores(x))
    _, bands = frontend.band_power(x)
    m = gate_mask(x, fe, bands)
    alarms = {"v4": [(t, "FIRE_ALARM") for t in frontend.alarm_times(x, cfg=cfg)]}
    flux = frontend.onset_strength(bands)
    ex = extras(fe, cat, m, alarms, lambda kappa, hits: frontend.knock_times(flux, kappa, hits))
    return dict(cat=cat, mask=m, f0=yamnet.detect(cat, siren_veto=True),
                f4=yamnet.detect(cat, m, ex, siren_veto=fe.get("veto", True)), alarms=alarms["v4"])


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        os.replace(path + ".part", path)
    return path


# --- DESED ----------------------------------------------------------------------------------

def desed_files():
    root = os.path.join(C.DATA, "desed")
    if not glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        tar = fetch(DESED_URL, os.path.join(root, "DESED_public_eval.tar.gz"))
        tarfile.open(tar).extractall(root)
        os.remove(tar)
    wavs = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)}
    events = defaultdict(list)
    for tsv in glob.glob(os.path.join(root, "**", "*.tsv"), recursive=True):
        rows = list(csv.DictReader(open(tsv), delimiter="\t"))
        if rows and {"filename", "onset", "offset", "event_label"} <= set(rows[0]):
            for r in rows:
                if r["onset"] not in ("", "NA"):
                    events[r["filename"]].append((float(r["onset"]), float(r["offset"]), r["event_label"]))
    return wavs, events


def desed_job(arg):
    path, evs, fe, cfg = arg
    x = streams.load(path)
    if len(x) < C.WIN:
        return None
    r = system(warm(x), fe, cfg)
    t_end = (C.WIN + np.arange(len(r["mask"])) * C.HOP) / C.SR - WARMUP_S  # window end, clip time
    t_start = t_end - C.WIN / C.SR
    keep = t_end > 0
    ev_out = []
    for on, off, label in evs:
        over = (t_end > on) & (t_start < off)
        ev_out.append(dict(label=label, awake=bool(r["mask"][over].any()) if over.any() else False))
    f0 = [(t - WARMUP_S, c) for t, c in r["f0"] if t > WARMUP_S]
    f4 = [(t - WARMUP_S, c) for t, c in r["f4"] if t > WARMUP_S]
    kept = sum(any(c4 == c and abs(t4 - t) <= 1.0 for t4, c4 in f4) for t, c in f0)
    return dict(duty=float(r["mask"][keep].mean()), events=ev_out, f0=len(f0), f0_kept=kept, f4=len(f4))


# --- SINS -----------------------------------------------------------------------------------

def sins_segments():
    """Labelled 10 s segments available in the downloaded SINS parts (the parts hold scattered
    segments, so episodes cannot be rebuilt): up to PER_ACTIVITY per activity, seeded."""
    root = os.path.join(C.DATA, "sins")
    for part in SINS_PARTS:
        z = os.path.join(root, f"{part}.zip")
        if not os.path.exists(z + ".done"):
            zipfile.ZipFile(fetch(SINS_URL.format(part=part), z)).extractall(root)
            os.remove(z)  # keep only the extracted files
            open(z + ".done", "w").close()
    meta = next(iter(glob.glob(os.path.join(root, "**", "meta.txt"), recursive=True)))
    have = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "**", "audio", "*.wav"), recursive=True)}
    by = defaultdict(list)
    for line in open(meta):
        f, label, _ = line.rstrip("\n").split("\t")
        if os.path.basename(f) in have:
            by[label].append(have[os.path.basename(f)])
    rng = np.random.default_rng(C.SEED)
    return [(label, p) for label, ps in sorted(by.items())
            for p in rng.permutation(sorted(ps))[:PER_ACTIVITY]]


def warm(x):
    """Prepend the first 2 s mirrored, so the noise floor settles before the scored audio."""
    return np.r_[x[:int(WARMUP_S * C.SR)][::-1], x]


def sins_job(arg):
    label, path, fe, cfg = arg
    x = streams.load(path)
    if len(x) < C.WIN:
        return None
    r = system(warm(x), fe, cfg)
    t_end = (C.WIN + np.arange(len(r["mask"])) * C.HOP) / C.SR
    keep = t_end > WARMUP_S
    return dict(label=label, hours=len(x) / C.SR / 3600, duty=float(r["mask"][keep].mean()),
                f0=[c for t, c in r["f0"] if t > WARMUP_S], f4=[c for t, c in r["f4"] if t > WARMUP_S],
                dsp=int(sum(t > WARMUP_S for t, _ in r["alarms"])))


def main():
    fe, cfg = proposed_gate(), frontend.chosen_alarm()
    if fe is None:
        print("  no results.json - run eval/evaluate.py first")
        return
    out = dict(gate=fe, alarm=cfg)
    with Pool(C.WORKERS) as pool:
        # DESED
        wavs, events = desed_files()
        res = [r for r in pool.map(desed_job, [(p, events.get(n, []), fe, cfg) for n, p in sorted(wavs.items())]) if r]
        by = defaultdict(list)
        for r in res:
            for e in r["events"]:
                by[e["label"]].append(e["awake"])
        f0, kept = sum(r["f0"] for r in res), sum(r["f0_kept"] for r in res)
        out["desed"] = dict(clips=len(res), duty=float(np.mean([r["duty"] for r in res])),
                            wake_recall={k: dict(n=len(v), rate=float(np.mean(v))) for k, v in sorted(by.items())},
                            wake_recall_all=float(np.mean([v for vs in by.values() for v in vs])),
                            f0_alerts=f0, f0_alerts_kept=kept, f4_alerts=sum(r["f4"] for r in res))
        print(f"\n  DESED: {len(res)} real clips, classifier duty {out['desed']['duty']:.1%}, gate awake for "
              f"{out['desed']['wake_recall_all']:.1%} of labelled events; always-on alerts kept {kept}/{f0}")
        for k, v in out["desed"]["wake_recall"].items():
            print(f"    {k:28s} n={v['n']:4d}  awake {v['rate']:.1%}")
        # SINS
        res = [r for r in pool.map(sins_job, [(lab, p, fe, cfg) for lab, p in sins_segments()]) if r]
    act = defaultdict(lambda: dict(hours=0.0, duty_h=0.0, f0=0, f4=0, dsp=0))
    for r in res:
        a = act[r["label"]]
        a["hours"] += r["hours"]
        a["duty_h"] += r["duty"] * r["hours"]
        a["f0"] += len(r["f0"])
        a["f4"] += len(r["f4"])
        a["dsp"] += r["dsp"]
    out["sins"] = {k: dict(hours=v["hours"], duty=v["duty_h"] / v["hours"], f0_per_h=v["f0"] / v["hours"],
                           f4_per_h=v["f4"] / v["hours"], dsp_alarms_per_h=v["dsp"] / v["hours"]) for k, v in act.items()}
    tot = sum(v["hours"] for v in act.values())
    out["sins_all"] = dict(segments=len(res), hours=tot, duty=sum(v["duty_h"] for v in act.values()) / max(tot, 1e-9),
                           f0_per_h=sum(v["f0"] for v in act.values()) / max(tot, 1e-9),
                           f4_per_h=sum(v["f4"] for v in act.values()) / max(tot, 1e-9),
                           f0_alerts=dict(zip(*np.unique([c for r in res for c in r["f0"]], return_counts=True))) if res else {})
    out["sins_all"]["f0_alerts"] = {k: int(v) for k, v in out["sins_all"]["f0_alerts"].items()}
    print(f"\n  SINS: {len(res)} segments, {tot:.1f} h of real home audio; classifier duty {out['sins_all']['duty']:.1%}; "
          f"alerts/h always-on {out['sins_all']['f0_per_h']:.2f} -> proposed {out['sins_all']['f4_per_h']:.2f}")
    for k, v in sorted(out["sins"].items(), key=lambda kv: -kv[1]["hours"]):
        print(f"    {k:16s} {v['hours']:5.1f} h  duty {v['duty']:6.1%}  alerts/h {v['f0_per_h']:5.2f} -> {v['f4_per_h']:5.2f}"
              f"  (DSP alarms/h {v['dsp_alarms_per_h']:.2f})")
    print(f"    always-on alerts by category: {out['sins_all']['f0_alerts']}")
    json.dump(out, open(os.path.join(C.RESULTS, "real_scenes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
