"""Step 1 (the slow part): per stream, render the audio, run YAMNet on EVERY window once (521
scores and the 1024-d embedding), compute the per-frame band powers (input of every gate variant)
and the DSP alarm detections, and cache them. Every front end is evaluated afterwards as a mask
over these cached outputs, which is exact because YAMNet scores each window independently:
skipping a window does not change any other window's output. The classifier head's training
streams are cached the same way (cache/v6train).
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

DIRS = {False: os.path.join(C.CACHE, "v6"), True: os.path.join(C.CACHE, "v6train")}


def job(arg):
    spec, train = arg
    path = os.path.join(DIRS[train], spec["id"] + ".npz")
    if os.path.exists(path):
        return path
    x, events = streams.render(spec)
    scores, emb = yamnet.window_outputs(x)
    level, bands = frontend.band_power(x)
    cfg = frontend.chosen_alarm()
    on = frontend.tone_track(x, cfg)
    np.savez_compressed(path + ".tmp.npz", scores=scores.astype(np.float16), emb=emb.astype(np.float16),
                        level=level.astype(np.float16), bands=bands.astype(np.float16), n=len(scores),
                        alarms=np.array(frontend.decide(on, cfg["rule"]), np.float32),
                        alarms_strict=np.array(frontend.t3_times(frontend.tone_track(x), strict=True), np.float32),
                        meta=json.dumps(dict(spec, events=events)))
    os.replace(path + ".tmp.npz", path)
    return path


def main(limit=None):
    for train in (True, False):
        os.makedirs(DIRS[train], exist_ok=True)
        specs = streams.specs(train)[:limit]
        t = time.time()
        with Pool(C.WORKERS) as pool:
            for i, _ in enumerate(pool.imap_unordered(job, [(sp, train) for sp in specs]), 1):
                if i % 20 == 0 or i == len(specs):
                    print(f"  {'training' if train else 'tuning+test'}: {i}/{len(specs)} streams, {time.time() - t:.0f} s",
                          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
