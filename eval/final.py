"""Step 3: the final system on audio outside the synthetic test streams.

Final system (all settings chosen on tuning data in eval/improve.py, "best_mn10" at a budget of
FINAL_BUDGET false alarms per hour per category): EfficientAT mn10 with the trained head per
category (score choice and smoothing), its thresholds, the two-floor gate, the Goertzel/T-3 alarm
detector, the knock detector and the siren veto. Compared with today's app (stock YAMNet, 0.30
thresholds, always on). Nothing is tuned here.
  real alarms  Freesound held-out smoke/fire alarms and look-alikes, clean and in a test room at 10 / 0 dB
  stress       alarm variants outside ISO 8201 and look-alike beepers (eval/stress.py)
  DESED        692 real 10 s home recordings with labelled events: gate wake recall, duty, alerts
  SINS         one real home, 10 s segments per activity: duty and alerts per hour
Audio is rendered by CPU workers (which also run YAMNet); EfficientAT runs on the GPU. Writes
results/final.json.
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import efficientat, head, yamnet  # noqa: E402
from dsp import frontend  # noqa: E402
from eval import evaluate as E  # noqa: E402
from eval import real_alarms as RA  # noqa: E402
from eval import real_scenes as RS  # noqa: E402
from eval import stress as ST  # noqa: E402
from eval.improve import smooth  # noqa: E402

SYSTEM, FINAL_BUDGET, MODEL = "best_mn10", "1.0", "mn10_as"
CFG = None


def settings():
    imp = json.load(open(os.path.join(C.RESULTS, "improve.json")))
    pt = imp["curve"][SYSTEM][FINAL_BUDGET]
    return dict(choice=imp["choices"][SYSTEM], thresholds=pt["thresholds"], fe=pt["full"]["frontend"])


def final_cat(scores, emb, choice):
    """Per-window category scores of the final classifier from mn10's outputs."""
    h = head.load(head.path_for(MODEL))
    base = {"stock": yamnet.category_scores(scores, head.labels_for(MODEL)), "head": head.category_scores(emb, scores, h)}
    base["mean"] = base["stock"].copy()
    base["mean"][:, :, 0] = (base["stock"][:, :, 0] + base["head"][:, :, 0]) / 2
    out = np.zeros_like(base["head"])
    for j, c in enumerate(C.ORDER):
        src, k = choice[c].split(":")[1].split("/")
        out[:, j] = smooth(base[src], int(k))[:, j]
    return out


def run_final(x, s, scores, emb):
    """Alerts, gate mask and DSP alarms of the final system on audio x."""
    fe = s["fe"]
    cat = final_cat(scores, emb, s["choice"])
    _, bands = frontend.band_power(x)
    m = RS.gate_mask(x, fe, bands)
    alarms = {"v4": [(t, "FIRE_ALARM") for t in frontend.alarm_times(x, cfg=CFG)]}
    flux = frontend.onset_strength(bands)
    ex = E.extras(fe, cat, m, alarms, lambda kappa, hits: frontend.knock_times(flux, kappa, hits))
    return yamnet.detect(cat, m, ex, s["thresholds"], siren_veto=True), m


def prep(item):
    """CPU worker: the audio and today's app (stock YAMNet always on) for one test item."""
    kind, key, x = item[0], item[1], item[2]
    if x is None:
        return None
    app = yamnet.detect(yamnet.category_scores(yamnet.window_scores(x)), siren_veto=True)
    return kind, key, x, app, item[3] if len(item) > 3 else None


def items():
    """(kind, key, audio, extra) for every test item; audio is built lazily by the workers."""
    out = []
    clips = [c for c in json.load(open(RA.MANIFEST)) if c["split"] == "heldout"]
    out += [("alarm", c, None, None) for c in clips]
    out += [("stress", (n, sd), None, None) for n in ST.VARIANTS for sd in range(ST.SEEDS)]
    wavs, events = RS.desed_files()
    out += [("desed", n, None, events.get(n, [])) for n in sorted(wavs)]
    out += [("sins", (lab, p), None, None) for lab, p in RS.sins_segments()]
    return out, wavs


def build(arg):
    kind, key, _, extra, wavs = arg
    from data import streams
    if kind == "alarm":
        return [prep(("alarm", (key, cond), y, None)) for cond, y in RA.conditions(key, pre_s=RA.PRE_S)]
    if kind == "stress":
        return [prep(("stress", key, ST.scene(*key), None))]
    if kind == "desed":
        x = streams.load(wavs[key])
        return [prep(("desed", key, RS.warm(x), extra))] if len(x) >= C.WIN else []
    x = streams.load(key[1])
    return [prep(("sins", key, RS.warm(x), len(x) / C.SR / 3600))] if len(x) >= C.WIN else []


def main():
    global CFG
    CFG = frontend.chosen_alarm()
    s = settings()
    print(f"  final system: {s['choice']}\n  thresholds {s['thresholds']}\n  gate {s['fe']}", flush=True)
    todo, wavs = items()
    res = defaultdict(list)
    with Pool(C.WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(build, [t + (wavs,) for t in todo], chunksize=4), 1):
            for r in out:
                if r is None:
                    continue
                kind, key, x, app, extra = r
                scores, emb = efficientat.window_outputs(x, MODEL)
                fired, m = run_final(x, s, scores, emb)
                res[kind].append((key, app, fired, m, extra))
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    out = dict(system=s)
    # real alarms: share of clips raising a fire alarm, per group and condition
    k0 = int(RA.PRE_S * C.SR) // C.HOP
    ra = defaultdict(lambda: defaultdict(list))
    for (clip, cond), app, fired, m, _ in res["alarm"]:
        g = ra[clip["group"]][cond]
        g.append(dict(app=any(c == "FIRE_ALARM" for _, c in app), final=any(c == "FIRE_ALARM" for _, c in fired),
                      awake=float(m[k0:].mean())))
    out["real_alarms"] = {g: {cond: dict(n=len(v), **{k: float(np.mean([r[k] for r in v])) for k in v[0]})
                              for cond, v in conds.items()} for g, conds in ra.items()}
    # stress
    st = defaultdict(list)
    for (name, _), app, fired, m, _ in res["stress"]:
        st[name].append(dict(app=any(c == "FIRE_ALARM" for _, c in app), final=any(c == "FIRE_ALARM" for _, c in fired)))
    out["stress"] = {n: dict(expected_fire=ST.VARIANTS[n][0], **{k: float(np.mean([r[k] for r in v])) for k in v[0]})
                     for n, v in st.items()}
    # DESED: wake recall per label, duty, alerts
    wake, duty, alerts = defaultdict(list), [], dict(app=defaultdict(int), final=defaultdict(int))
    for name, app, fired, m, evs in res["desed"]:
        t_end = (C.WIN + np.arange(len(m)) * C.HOP) / C.SR - RS.WARMUP_S
        t_start = t_end - C.WIN / C.SR
        duty.append(float(m[t_end > 0].mean()))
        for on, off, label in evs:
            over = (t_end > on) & (t_start < off)
            wake[label].append(bool(m[over].any()) if over.any() else False)
        for key, al in (("app", app), ("final", fired)):
            for t, c in al:
                if t > RS.WARMUP_S:
                    alerts[key][c] += 1
    out["desed"] = dict(clips=len(res["desed"]), duty=float(np.mean(duty)),
                        wake_recall={k: dict(n=len(v), rate=float(np.mean(v))) for k, v in sorted(wake.items())},
                        wake_recall_all=float(np.mean([v for vs in wake.values() for v in vs])),
                        alerts={k: dict(v) for k, v in alerts.items()})
    # SINS: duty and alerts per hour per activity
    act = defaultdict(lambda: dict(hours=0.0, duty_h=0.0, app=0, final=0))
    for (label, _), app, fired, m, hrs in res["sins"]:
        t_end = (C.WIN + np.arange(len(m)) * C.HOP) / C.SR
        a = act[label]
        a["hours"] += hrs
        a["duty_h"] += float(m[t_end > RS.WARMUP_S].mean()) * hrs
        a["app"] += sum(t > RS.WARMUP_S for t, _ in app)
        a["final"] += sum(t > RS.WARMUP_S for t, _ in fired)
    tot = sum(a["hours"] for a in act.values())
    out["sins"] = {k: dict(hours=a["hours"], duty=a["duty_h"] / a["hours"], app_per_h=a["app"] / a["hours"],
                           final_per_h=a["final"] / a["hours"]) for k, a in act.items()}
    out["sins_all"] = dict(hours=tot, duty=sum(a["duty_h"] for a in act.values()) / tot,
                           app_per_h=sum(a["app"] for a in act.values()) / tot,
                           final_per_h=sum(a["final"] for a in act.values()) / tot)
    out["macs"] = dict(yamnet=yamnet.macs_per_window(), mn10=efficientat.macs_per_window(MODEL),
                       head=head.macs(head.load(head.path_for(MODEL))))
    json.dump(out, open(os.path.join(C.RESULTS, "final.json"), "w"), indent=1)
    print("\n  real alarms (share raising a fire alarm; awake = classifier duty during the clip):")
    for g, conds in out["real_alarms"].items():
        for cond, v in conds.items():
            print(f"    {g:12s} {cond:6s} n={v['n']:3d}  app {v['app']:.2f}  final {v['final']:.2f}  awake {v['awake']:.2f}")
    print("  stress:")
    for n, v in out["stress"].items():
        print(f"    {n:30s} should fire {'yes' if v['expected_fire'] else 'no ':3s}  app {v['app']:.2f}  final {v['final']:.2f}")
    d = out["desed"]
    print(f"  DESED: {d['clips']} clips, duty {d['duty']:.1%}, wake recall {d['wake_recall_all']:.1%}, alerts {d['alerts']}")
    sa = out["sins_all"]
    print(f"  SINS: {sa['hours']:.1f} h, duty {sa['duty']:.1%}, alerts/h app {sa['app_per_h']:.2f} final {sa['final_per_h']:.2f}")
    for k, v in sorted(out["sins"].items(), key=lambda kv: -kv[1]["hours"]):
        print(f"    {k:16s} {v['hours']:5.2f} h duty {v['duty']:6.1%} alerts/h app {v['app_per_h']:.2f} final {v['final_per_h']:.2f}")
    print(f"  MACs per window: {out['macs']}")


if __name__ == "__main__":
    main()
