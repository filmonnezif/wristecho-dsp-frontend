"""Step 1 (the slow part): per stream, render the audio, run YAMNet on EVERY window once, compute
the gate features and the DSP alarm detections, and cache them. Every front end is evaluated
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
    path = os.path.join(C.CACHE, spec["id"] + ".npz")
    if os.path.exists(path):
        return path
    x, events = streams.render(spec)
    n = yamnet.n_windows(len(x))
    t0 = time.process_time()
    cat = yamnet.category_scores(yamnet.window_scores(x))
    t_yam = (time.process_time() - t0) / n
    t0 = time.process_time()
    level, novelty = frontend.gate_features(x)
    lvl_w, nov_w = frontend.per_window(level, n), frontend.per_window(novelty, n)
    t_gate = (time.process_time() - t0) / n
    t0 = time.process_time()
    alarms = frontend.alarm_times(x)
    t_alarm = (time.process_time() - t0) / n
    np.savez_compressed(path + ".tmp.npz", cat=cat.astype(np.float16), level=lvl_w, novelty=nov_w,
                        alarms=np.array(alarms, np.float32), cost=np.array([t_yam, t_gate, t_alarm]),
                        meta=json.dumps(dict(spec, events=events)))
    os.replace(path + ".tmp.npz", path)
    return path


def main(limit=None):
    os.makedirs(C.CACHE, exist_ok=True)
    specs = streams.specs()[:limit]
    t = time.time()
    with Pool(C.WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, specs), 1):
            if i % 20 == 0 or i == len(specs):
                print(f"  {i}/{len(specs)} streams, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
