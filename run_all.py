"""Reproduce the whole study: python run_all.py  (about 1 h on a Colab CPU runtime).

  1. setup/download.py     datasets + model (skips what exists)
  1b. eval/alarm_design.py redesign the alarm detector on the design half of the real recordings
  2. eval/extract.py       classifier scores, gate features, alarm detections per stream (cached)
  3. eval/evaluate.py      calibrate on tuning environments, evaluate on unseen test environments
     eval/stress.py        alarm detector outside its specification and on look-alike sounds
     eval/real_alarms.py   real Freesound alarms and look-alikes (clip list in data/freesound.json)
     eval/real_scenes.py   the proposed system on real recordings: DESED (labelled events) and SINS (home life)
  4. figures/make.py       every figure -> figures/out/
Optional argument: number of streams (a quick smoke test, e.g. python run_all.py 4).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import alarm_design, evaluate, extract, real_alarms, real_scenes, stress  # noqa: E402
from figures import make  # noqa: E402
from paper import make_tables as tables  # noqa: E402
from setup import download  # noqa: E402


def step(name, fn, *a):
    t = time.time()
    print(f"[{name}]", flush=True)
    fn(*a)
    print(f"[{name}] {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    step("download", download.main)
    step("alarm design", alarm_design.main)
    step("extract", extract.main, limit)
    step("evaluate", evaluate.main)
    step("stress", stress.main)
    step("real alarms", real_alarms.main)
    step("real scenes", real_scenes.main)
    step("figures", make.main)
    step("tables", tables.main)
