"""Reproduce the whole study: python run_all.py  (about 1 h on a Colab CPU runtime).

  1. setup/download.py     datasets + model (skips what exists)
  2. eval/extract.py       classifier scores, gate features, alarm detections per stream (cached)
  3. eval/evaluate.py      calibrate on tuning environments, evaluate on unseen test environments
     eval/stress.py        alarm detector outside its specification and on look-alike sounds
  4. figures/make.py       every figure -> figures/out/
Optional argument: number of streams (a quick smoke test, e.g. python run_all.py 4).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import evaluate, extract, stress  # noqa: E402
from figures import make  # noqa: E402
from setup import download  # noqa: E402


def step(name, fn, *a):
    t = time.time()
    print(f"[{name}]", flush=True)
    fn(*a)
    print(f"[{name}] {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    step("download", download.main)
    step("extract", extract.main, limit)
    step("evaluate", evaluate.main)
    step("stress", stress.main)
    step("figures", make.main)
