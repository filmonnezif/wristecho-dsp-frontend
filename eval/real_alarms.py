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
MAX_S, SNRS = 30.0, [None, 10, 0]


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


def conditions(clip):
    """[(condition, audio)]: the clip's first 30 s clean, and mixed into a test-environment
    background at 10 and 0 dB SNR (over the clip's active frames)."""
    x = load_mp3(os.path.join(FOLDER, f"{clip['id']}.mp3"))[:int(MAX_S * C.SR)]
    if len(x) < C.WIN:
        return []
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    _, _, rms = streams.active(x)
    rng = np.random.default_rng(clip["id"])
    env = C.TEST_ENVS[clip["id"] % len(C.TEST_ENVS)]
    bg = streams.load(os.path.join(C.DATA, "demand", env, "ch01.wav"))
    i = int(rng.integers(0, len(bg) - len(x)))
    bg = bg[i:i + len(x)] / (np.sqrt(np.mean(bg[i:i + len(x)] ** 2)) + 1e-9)
    out = []
    for snr in SNRS:
        y = x if snr is None else (x / rms + bg * 10 ** (-snr / 20)) * rms
        out.append(("clean" if snr is None else f"{snr} dB", np.clip(y, -1, 1).astype(np.float32)))
    return out


def job(arg):
    clip, thresholds, chosen = arg
    out = {}
    for cond, y in conditions(clip):
        cat = yamnet.category_scores(yamnet.window_scores(y))
        a3 = bool(frontend.alarm_times(y, strict=True))
        a4 = bool(frontend.alarm_times(y, cfg=chosen))
        f0 = any(c == "FIRE_ALARM" for _, c in yamnet.detect(cat, siren_veto=True))
        out[cond] = dict(F0=f0, A_v3=a3, A_v4=a4, F0_A_v3=f0 or a3, F0_A_v4=f0 or a4)
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
    with Pool(C.WORKERS) as pool:
        res = [r for r in pool.map(job, [(c, None, chosen) for c in clips]) if r]
    summary = {}
    for split in ("design", "heldout"):
        for group in QUERIES:
            rs = [o for c, o in res if c["group"] == group and c["split"] == split]
            if rs:
                summary[f"{group} | {split}"] = dict(n=len(rs), fire_expected=QUERIES[group][0],
                                                      **{cond: {d: float(np.mean([o[cond][d] for o in rs])) for d in rs[0][cond]}
                                                         for cond in rs[0]})
    out = dict(chosen=chosen, summary=summary, clips=[dict(c, result=o) for c, o in res])
    json.dump(out, open(os.path.join(C.RESULTS, "real_alarms.json"), "w"), indent=1)
    print(f"\n  detector v4: {chosen}")
    print(f"  {'group | split':26s} {'n':>3s} {'cond':>7s} {'F0':>6s} {'DSP v3':>7s} {'DSP v4':>7s} {'F0+v3':>6s} {'F0+v4':>6s}")
    for g, s in summary.items():
        for cond in ("clean", "10 dB", "0 dB"):
            v = s[cond]
            print(f"  {g:26s} {s['n']:3d} {cond:>7s} {v['F0']:6.2f} {v['A_v3']:7.2f} {v['A_v4']:7.2f} "
                  f"{v['F0_A_v3']:6.2f} {v['F0_A_v4']:6.2f}")
    print("  (share of clips raising a fire alarm; F0 = YAMNet with siren veto; for look-alikes lower is better)")


if __name__ == "__main__":
    search() if sys.argv[1:] == ["search"] else main()
