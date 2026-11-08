"""Reproduce the whole study: python run_all.py  (about 40 min on a Colab runtime with many CPU cores;
EfficientAT runs on the GPU if there is one).

  1. setup/download.py      datasets + model (skips what exists); setup/extra_data.py: extra training clips
  1b. eval/alarm_design.py  the alarm detector, designed on the design half of the real recordings
  2. eval/extract.py        YAMNet scores + embeddings, gate features, alarm detections per stream (cached)
     eval/extract_eat.py    EfficientAT outputs for the same windows (GPU)
     eval/train_head.py     the classifier heads on frozen embeddings (training streams)
  3. eval/evaluate.py       gates, DSP detectors, stock YAMNet: calibrated on tuning, tested on unseen environments
     eval/improve.py        per-class score choice, smoothing, thresholds under false-alarm budgets; the curve
     eval/final.py          the final systems on real alarms, stress variants, DESED and SINS
     eval/predominant.py    recall on the predominant-target subset
     eval/export.py         paper/data/results_v7.json
  4. figures/make.py        every figure -> figures/out/
Optional argument: number of streams (a quick smoke test, e.g. python run_all.py 4).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import alarm_design, evaluate, export, extract, extract_eat, final, improve, predominant, train_head  # noqa: E402
from figures import make  # noqa: E402
from setup import download, extra_data  # noqa: E402


def step(name, fn, *a):
    t = time.time()
    print(f"[{name}]", flush=True)
    fn(*a)
    print(f"[{name}] {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    step("download", download.main)
    step("extra training data", extra_data.main)
    step("alarm design", alarm_design.main)
    step("extract", extract.main, limit)
    step("extract EfficientAT", extract_eat.main)
    step("train heads", train_head.main)
    step("evaluate", evaluate.main)
    step("improve", improve.main)
    step("final", final.main)
    step("predominant", predominant.main)
    step("export", export.main)
    step("figures", make.main)
