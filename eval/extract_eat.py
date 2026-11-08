"""Step 1d: EfficientAT outputs for every stream (tuning, test and training), cached next to YAMNet's.
Streams are rendered on the CPU workers and classified on the GPU. cache/eat_<model>/<stream>.npz
holds scores (527) and embeddings per window, for the same windows as YAMNet's cache.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import efficientat  # noqa: E402
from data import streams  # noqa: E402


def folder(name):
    return os.path.join(C.CACHE, f"eat_{name}")


def render(spec):
    return spec["id"], streams.render(spec)[0]


def main():
    specs = streams.specs() + streams.specs(True)
    import json
    for name in efficientat.MODELS:
        os.makedirs(folder(name), exist_ok=True)
        json.dump(efficientat.labels(name), open(os.path.join(folder(name), "labels.json"), "w"))
        missing = [l for c in C.CATEGORIES.values() for l in c["labels"] if l not in efficientat.labels(name)]
        print(f"  {name}: category labels missing from its class list: {missing or 'none'}", flush=True)
    todo = [s for s in specs if any(not os.path.exists(os.path.join(folder(m), s["id"] + ".npz")) for m in efficientat.MODELS)]
    t = time.time()
    with Pool(C.WORKERS) as pool:
        for i, (sid, x) in enumerate(pool.imap_unordered(render, todo), 1):
            for name in efficientat.MODELS:
                scores, emb = efficientat.window_outputs(x, name)
                np.savez_compressed(os.path.join(folder(name), sid + ".npz"), scores=scores.astype(np.float16),
                                    emb=emb.astype(np.float16))
            if i % 50 == 0 or i == len(todo):
                print(f"  EfficientAT: {i}/{len(todo)} streams, {time.time() - t:.0f} s", flush=True)
    for name in efficientat.MODELS:
        print(f"  {name}: {efficientat.macs_per_window(name):,} MACs per window, embedding "
              f"{np.load(os.path.join(folder(name), specs[0]['id'] + '.npz'))['emb'].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
