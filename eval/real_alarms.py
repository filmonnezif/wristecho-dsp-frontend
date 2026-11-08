"""Step 2c: real recordings from Freesound, to check the synthetic-alarm results.

Clips are split by uploader into a design half (used only by eval/alarm_design.py to redesign the
detector) and a held-out half (reported). Positives: recordings uploaded as smoke alarms / smoke
detectors / fire alarms. Negatives:
look-alikes that should NOT raise a fire alarm (microwave and oven beeps, reversing beepers, alarm
clocks). Low-battery chirps are excluded (they are not alarms).

  python eval/real_alarms.py search    # authors only, once: needs a Freesound API key in
                                        # ~/.config/wristecho/freesound.key; writes data/freesound.json
  python eval/real_alarms.py           # downloads the listed previews (no key needed) and evaluates

Each clip (first 30 s, 16 kHz mono) is tested clean and mixed into a test-environment background
at 10 and 0 dB SNR. Detectors: the classifier with the siren veto (F0), the first DSP detector (v3),
the strict T-3 detector, and F0+A (classifier or T-3, the proposed system's alarm decision).
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402

MANIFEST = os.path.join(C.ROOT, "data", "freesound.json")
FOLDER = os.path.join(C.DATA, "freesound")
QUERIES = {  # group: (should raise a fire alarm?, queries)
    "smoke alarm": (True, ["smoke alarm", "smoke detector", "smoke alarm beeping", "fire alarm beeping", "smoke alarm going off",
                           "detector alarm", "alarm beeping house"]),
    "fire alarm": (True, ["fire alarm"]),
    "look-alike": (False, ["microwave beep", "oven timer beep", "reversing beeper", "alarm clock beep", "kitchen timer beep",
                           "truck backing up beep", "washing machine beep", "heart monitor beep", "phone beep", "pager beep"]),
}
PER_QUERY = {"smoke alarm": 40, "fire alarm": 30, "look-alike": 12}
EXCLUDE_WORDS = ("battery", "chirp", "test button", "music", "song", "remix", "carbon monoxide", "co alarm")
POSITIVE_WORDS = ("smoke", "alarm", "detector")  # a positive's name must say what it is
MAX_S, SNRS, PRE_S = 30.0, [None, 10, 0], 5.0
# The classes added in v6, in a manifest of their own (data/freesound_extra.json) so the alarm design
# above keeps its clips: baby cries (stream targets: the design half goes to the tuning/training
# clips, the held-out half to the test clips).
EXTRA_MANIFEST = os.path.join(C.ROOT, "data", "freesound_extra.json")
EXTRA_QUERIES = {  # group: (queries, the name must contain one of, and one of, clips per query)
    "baby cry": (["baby crying", "baby cry", "infant crying", "newborn crying", "toddler crying", "baby screaming"],
                 ("baby", "infant", "newborn", "toddler", "child", "kid"), ("cry", "cries", "crying", "scream", "wail", "sob"), 60),
}
EXTRA_EXCLUDE = ("music", "song", "remix", "laugh", "giggl", "toy", "doll", "fake", "imitat", "cartoon", "synth", "loop",
                 "game", "slowed", "joy", "children", "monitor")


def search_extra():
    key = open(os.path.expanduser("~/.config/wristecho/freesound.key")).read().strip()
    seen, out = {c["id"] for c in json.load(open(MANIFEST))}, []
    for group, (queries, must1, must2, per_query) in EXTRA_QUERIES.items():
        for q in queries:
            params = dict(query=q, filter="duration:[2 TO 120]", sort="score", page_size=150,
                          fields="id,name,tags,license,username,duration,previews,description")
            req = urllib.request.Request("https://freesound.org/apiv2/search/text/?" + urllib.parse.urlencode(params),
                                         headers={"Authorization": f"Token {key}"})
            n = 0
            for r in json.load(urllib.request.urlopen(req, timeout=60))["results"]:
                name = r["name"].lower().replace("_", " ")
                text = name + " " + " ".join(r["tags"]).lower() + " " + r.get("description", "").lower()
                if r["id"] in seen or "Sampling" in r["license"] or any(w in text for w in EXTRA_EXCLUDE):
                    continue
                if not (any(w in name for w in must1) and any(w in name for w in must2)):
                    continue
                seen.add(r["id"])
                split = "design" if zlib.crc32(r["username"].encode()) % 2 == 0 else "heldout"
                out.append(dict(id=r["id"], group=group, query=q, split=split, name=r["name"], user=r["username"],
                                license=r["license"], duration=r["duration"], preview=r["previews"]["preview-hq-mp3"]))
                n += 1
                if n >= per_query:
                    break
    json.dump(out, open(EXTRA_MANIFEST, "w"), indent=1)
    from collections import Counter
    print(f"  {len(out)} clips -> {EXTRA_MANIFEST}: {Counter((c['group'], c['split']) for c in out)}")


def search():
    key = open(os.path.expanduser("~/.config/wristecho/freesound.key")).read().strip()
    seen, out = set(), []
    for group, (fire, queries) in QUERIES.items():
        for q in queries:
            params = dict(query=q, filter="duration:[3 TO 180]", sort="score", page_size=150,
                          fields="id,name,tags,license,username,duration,previews,description")
            req = urllib.request.Request("https://freesound.org/apiv2/search/text/?" + urllib.parse.urlencode(params),
                                         headers={"Authorization": f"Token {key}"})  # key in a header, not the URL
            res = json.load(urllib.request.urlopen(req, timeout=60))["results"]
            n = 0
            for r in res:
                text = (r["name"] + " " + " ".join(r["tags"]) + " " + r.get("description", "")).lower()
                if r["id"] in seen or any(w in text for w in EXCLUDE_WORDS) or "Sampling" in r["license"]:
                    continue
                if fire and not any(w in r["name"].lower() for w in POSITIVE_WORDS):
                    continue
                seen.add(r["id"])
                # design / held-out split by uploader: no recording device or room is in both halves
                split = "design" if zlib.crc32(r["username"].encode()) % 2 == 0 else "heldout"
                out.append(dict(id=r["id"], group=group, query=q, fire=fire, split=split, name=r["name"], user=r["username"],
                                license=r["license"], duration=r["duration"], preview=r["previews"]["preview-hq-mp3"]))
                n += 1
                if n >= PER_QUERY[group]:
                    break
    json.dump(out, open(MANIFEST, "w"), indent=1)
    print(f"  {len(out)} clips -> {MANIFEST}")


def load_mp3(path):
    try:
        return streams.load(path)
    except Exception:  # noqa: BLE001  (older libsndfile without MP3): decode with ffmpeg
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(C.SR), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).copy()


def fetch(clip):
    path = os.path.join(FOLDER, f"{clip['id']}.mp3")
    if not os.path.exists(path):
        with urllib.request.urlopen(clip["preview"], timeout=60) as r:
            open(path, "wb").write(r.read())
    return path


def conditions(clip, pre_s=0.0):
    """[(condition, audio)]: the clip's first 30 s clean, and mixed into a test-environment
    background at 10 and 0 dB SNR (over the clip's active frames). pre_s: seconds of that
    background (40 dB below the clip for "clean") before the clip, so the gate's noise floor has
    settled on the room before the alarm starts, as on an always-listening watch."""
    x = load_mp3(os.path.join(FOLDER, f"{clip['id']}.mp3"))[:int(MAX_S * C.SR)]
    if len(x) < C.WIN:
        return []
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    _, _, rms = streams.active(x)
    rng = np.random.default_rng(clip["id"])
    env = C.TEST_ENVS[clip["id"] % len(C.TEST_ENVS)]
    bg = streams.load(os.path.join(C.DATA, "demand", env, "ch01.wav"))
    n_pre = int(pre_s * C.SR)
    i = int(rng.integers(0, len(bg) - len(x) - n_pre))
    bg = bg[i:i + n_pre + len(x)]
    bg = bg / (np.sqrt(np.mean(bg ** 2)) + 1e-9)
    out = []
    for snr in SNRS:
        g = 10 ** (-(40 if snr is None else snr) / 20)
        y = np.r_[bg[:n_pre] * g, (x / rms + (0 if snr is None else bg[n_pre:] * g))] * rms
        out.append(("clean" if snr is None else f"{snr} dB", np.clip(y, -1, 1).astype(np.float32)))
    return out


def job(arg):
    clip, gates, chosen = arg
    from eval.real_scenes import outputs, system
    out = {}
    for cond, y in conditions(clip, pre_s=PRE_S):
        cats = outputs(y)
        a3 = frontend.alarm_times(y, strict=True)
        a4 = frontend.alarm_times(y, cfg=chosen)
        f0 = any(c == "FIRE_ALARM" for _, c in yamnet.detect(cats["stock"], siren_veto=True))
        r = dict(F0=f0, A_v3=bool(a3), A_v4=bool(a4), F0_A_v3=f0 or bool(a3), F0_A_v4=f0 or bool(a4))
        k0 = int(PRE_S * C.SR) // C.HOP  # first window that reaches into the clip
        for clf, fe in gates.items():  # the proposed system: classifier only while the gate is awake, plus DSP
            sy = system(y, fe, chosen, cats)
            tag = "F4" if clf == "stock" else "H4"
            r[tag] = any(c == "FIRE_ALARM" for _, c in sy["f4"])
            if clf == "head":
                r["H0"] = any(c == "FIRE_ALARM" for _, c in sy["f0"])
            r[f"awake_during_{clf}"] = float(sy["mask"][k0:].mean())
            r[f"awake_before_{clf}"] = float(sy["mask"][:max(k0 - 2, 1)].mean())
        out[cond] = r
    return (clip, out) if out else None


def clips_downloaded():
    clips = json.load(open(MANIFEST))
    os.makedirs(FOLDER, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch, clips))
    return clips


def main():
    if not os.path.exists(MANIFEST):
        print("  no data/freesound.json (run: python eval/real_alarms.py search) - skipped")
        return
    clips = clips_downloaded()
    chosen = frontend.chosen_alarm()
    from eval.real_scenes import proposed_gate
    gates = {clf: fe for clf in ("stock", "head") if (fe := proposed_gate(clf=clf))}
    with Pool(C.WORKERS) as pool:
        res = [r for r in pool.map(job, [(c, gates, chosen) for c in clips]) if r]
    summary = {}
    for split in ("design", "heldout"):
        for group in QUERIES:
            rs = [o for c, o in res if c["group"] == group and c["split"] == split]
            if rs:
                summary[f"{group} | {split}"] = dict(n=len(rs), fire_expected=QUERIES[group][0],
                                                      **{cond: {d: float(np.mean([o[cond][d] for o in rs])) for d in rs[0][cond]}
                                                         for cond in rs[0]})
    out = dict(chosen=chosen, gates=gates, summary=summary, clips=[dict(c, result=o) for c, o in res])
    json.dump(out, open(os.path.join(C.RESULTS, "real_alarms.json"), "w"), indent=1)
    print(f"\n  detector v4: {chosen}")
    print(f"  gates (proposed): {gates}")
    print(f"  {'group | split':26s} {'n':>3s} {'cond':>7s} {'F0':>6s} {'DSP v4':>7s} {'F0+v4':>6s} {'F4':>6s}"
          f" {'H0':>6s} {'H4':>6s} {'awake':>6s} {'before':>6s}")
    nan = float("nan")
    for g, s in summary.items():
        for cond in ("clean", "10 dB", "0 dB"):
            v = s[cond]
            print(f"  {g:26s} {s['n']:3d} {cond:>7s} {v['F0']:6.2f} {v['A_v4']:7.2f} {v['F0_A_v4']:6.2f} "
                  f"{v.get('F4', nan):6.2f} {v.get('H0', nan):6.2f} {v.get('H4', nan):6.2f} "
                  f"{v.get('awake_during_head', v.get('awake_during_stock', nan)):6.2f} "
                  f"{v.get('awake_before_head', v.get('awake_before_stock', nan)):6.2f}")
    print("  (share of clips raising a fire alarm; F0/H0 = stock YAMNet / trained head on every window with siren veto;")
    print("   F4/H4 = the proposed system with each; awake/before = classifier duty during the clip / the lead-in)")


if __name__ == "__main__":
    {"search": search, "search_extra": search_extra}.get(sys.argv[1] if sys.argv[1:] else "", main)()
