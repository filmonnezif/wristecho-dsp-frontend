# Listen Before You Classify

**An adaptive DSP front end that wakes a sound classifier only when needed and detects emergency alarms by itself.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/filmonnezif/wristecho-dsp-frontend/blob/main/colab.ipynb)

Wearable sound-awareness devices for Deaf and hard-of-hearing people run a neural classifier
(here YAMNet, 70.7 M multiply-accumulates per 0.975 s window) on every half second of audio, even
though nothing happens most of the time, and pretrained classifiers are weak on standardized
emergency signals such as the ISO 8201 T-3 smoke-alarm pattern.

**Research question.** Can an adaptive DSP front end that detects emergency alarms itself and wakes
the classifier only when a new sound starts cut classifier computation substantially while raising
alarm recall, in acoustic environments it was never tuned on?

```
mic 16 kHz ─┬─ HPF (Butterworth, 100 Hz) → 20 ms FFT → 16 bands → noise-floor tracker → novelty ─→ wake YAMNet ─→ alert
            └─ Goertzel resonator bank (2.8–3.6 kHz, 520 Hz + harmonics) → T-3 pattern matcher ───────────────→ alert
```

## Reproduce

One command, about an hour on a Colab CPU runtime (downloads included):

```sh
pip install numpy scipy soundfile matplotlib ai-edge-litert pytest
python -m pytest tests        # controlled DSP test cases
python run_all.py             # download → extract → evaluate → figures
python run_all.py 8           # quick smoke test on 8 streams
```

| Step | What it does | Output |
|---|---|---|
| `setup/download.py` | YAMNet, DEMAND (17 environments), ESC-50, the FSD50K clips in `data/cliplist.json` | `datasets/` |
| `eval/extract.py` | per stream: classifier scores on **every** window, gate features, alarm detections | `cache/*.npz` |
| `eval/evaluate.py` | front ends as masks over the cached scores; calibrate on tuning environments, report on unseen ones | `results/results.json` |
| `figures/make.py` | all figures | `figures/out/` |

All parameters are in [`config.py`](config.py), with the reason for each value.

## Data

Public datasets only. **Backgrounds:** DEMAND, 11 environments for tuning and 6 unseen ones for
testing (one per category). **Targets:** sirens, doorbells and knocks from FSD50K and ESC-50 (official
splits: dev → tune, eval → test), plus smoke alarms synthesized to ISO 8201 (piezo ~3.1 kHz and the
520 Hz low-frequency square wave), since no public set of real smoke-alarm recordings exists.
**Distractors:** everyday FSD50K/ESC-50 sounds, half of them easily confused with the targets.
Streams are 5 min long with sparse events, following the event-mixing protocol of DCASE 2017
Task 2, plus "idle" streams of background only.

## Front ends compared

| | Front end |
|---|---|
| F0 | classifier on every window (baseline) |
| F1 | classifier on every k-th window (duty cycling) |
| F2 | fixed energy gate |
| F3 | adaptive gate (novelty above a tracked noise floor) |
| F4 | F3 + Goertzel/T-3 alarm detector (**proposed**) |
| F0+A | F0 + alarm detector (ablation) |

Gate thresholds are the lowest-compute settings that lose at most 1 % of the baseline's detections
on the tuning environments; every number in the paper is measured on the test environments.

## Repository

```
config.py  run_all.py  colab.ipynb
setup/       downloads
data/        clip list, stream builder, ISO 8201 alarm synthesis
dsp/         HPF, noise-floor tracker, novelty gate, Goertzel bank, T-3 matcher
classifier/  YAMNet runner and the alert decision rule
eval/        extraction, calibration, metrics, statistics
figures/     paper figures
tests/       unit tests for the DSP blocks
paper/       IEEE paper
demo/        live MATLAB demo (coming)
```
