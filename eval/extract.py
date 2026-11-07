"""Step 1 (the slow part): per stream, render the audio, run YAMNet on EVERY window once, compute
the per-frame band powers (input of every gate variant) and the DSP alarm detections, and cache them. Every front end is evaluated
afterwards as a mask over these cached scores, which is exact because YAMNet scores each window
independently: skipping a window does not change any other window's score.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402


def job(spec):
    path = os.path.join(C.CACHE, "v2", spec["id"] + ".npz")
    if os.path.exists(path):
        return path
    x, events = streams.render(spec)
    n = yamnet.n_windows(len(x))
    v1 = os.path.join(C.CACHE, spec["id"] + ".npz")  # first version's cache: same audio, same scores
    if os.path.exists(v1):
        cat = np.load(v1)["cat"]
    else:
        cat = yamnet.category_scores(yamnet.window_scores(x)).astype(np.float16)
    level, bands = frontend.band_power(x)
    on = frontend.tone_track(x)
    np.savez_compressed(path + ".tmp.npz", cat=cat, level=level.astype(np.float16), bands=bands.astype(np.float16),
                        n=n, alarms=np.array(frontend.t3_times(on), np.float32),
                        alarms_strict=np.array(frontend.t3_times(on, strict=True), np.float32),
                        meta=json.dumps(dict(spec, events=events)))
    os.replace(path + ".tmp.npz", path)
    return path


def alarm_dir(cfg):
    return os.path.join(C.CACHE, "alarms_" + "_".join(str(cfg[k]) for k in ("lo", "hi", "ratio", "rule")).replace("|", "+"))


def alarm_job(arg):
    """Fire-alarm times of the redesigned detector for one stream (cached per configuration)."""
    spec, cfg = arg
    path = os.path.join(alarm_dir(cfg), spec["id"] + ".npy")
    if not os.path.exists(path):
        x, _ = streams.render(spec)
        np.save(path, np.array(frontend.alarm_times(x, cfg=cfg), np.float32))
    return path


def main(limit=None):
    os.makedirs(os.path.join(C.CACHE, "v2"), exist_ok=True)
    specs = streams.specs()[:limit]
    t = time.time()
    with Pool(C.WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, specs), 1):
            if i % 20 == 0 or i == len(specs):
                print(f"  {i}/{len(specs)} streams, {time.time() - t:.0f} s", flush=True)
        cfg = frontend.chosen_alarm()
        os.makedirs(alarm_dir(cfg), exist_ok=True)
        pool.map(alarm_job, [(sp, cfg) for sp in specs])
        print(f"  alarm detector {cfg}: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
