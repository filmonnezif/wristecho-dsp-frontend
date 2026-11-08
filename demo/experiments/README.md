# Experiments run with the demo's exact front end

## gainSirenTest.m: does a fixed gain help low-level sirens wake the gate?
"17 dB" is the gate threshold Δ at the relaxed budget: how far a band's level must rise above that band's own
tracked noise floor (relative dB), not dBFS, SNR or a gain. Gain (0, +3, +6, +12 dB) is applied to every sample
of the input (sound + room) before the front end; recordings, classifier (EfficientAT + head) and thresholds
(relaxed budget, Δ = 17 dB, H = 0) are identical. 8 ESC-50 test-fold sirens at 10 / 5 / 0 dB SNR (24 events), 4
negatives at 10 dB. Run in MATLAB Online on 2026-11-08 (151 s); results in `gainSirenTest.mat` (MATLAB Drive).

| Gain | Trigger coverage | Siren recall | Gate wakes outside sirens | Gate wakes, negatives | False alerts, negatives | Classifier calls | Clipped |
|---|---|---|---|---|---|---|---|
| +0 dB | 49.2 % | 25.0 % (6/24) | 0 | 55 | 2 | 185 | 0 % |
| +3 dB | 49.2 % | 25.0 % | 0 | 55 | 1 | 185 | 0 % |
| +6 dB | 49.2 % | 29.2 % | 0 | 55 | 1 | 185 | 0 % |
| +12 dB | 49.2 % | 41.7 % (10/24) | 0 | 55 | 1 | 185 | 0 % |

By SNR, recall at 0 / +3 / +6 / +12 dB: 10 dB 50 / 50 / 63 / 63 %, 5 dB 13 / 13 / 13 / 50 %, 0 dB 13 % at every
gain; trigger coverage 92 % (10 dB), 51 % (5 dB), 5 % (0 dB) at every gain.

**Reading.** The gain does not change the trigger at all (identical coverage, wakes and classifier calls): the
gate compares each band with its own floor, so a constant gain cancels. It does change the classifier: the log-mel
front end adds a fixed 1e-5 before the log, so a quiet input sits closer to that floor than the louder clips the
network was trained on; lifting the level made the head more confident on sirens it already saw. No extra
processing, no clipping (the scenes peak far below 0 dBFS), and no more false alerts here. Sirens at 0 dB SNR stay
missed: the gate never wakes for them, and no gain can fix that.

**Recommendation.** +12 dB was best here, but on 24 events (10 vs. 6 detections) and only as an input-level
effect on the classifier. Treat it as a hypothesis: normalise the classifier's input level (not the gate's), and
validate on the full benchmark and the real-home recordings (SINS) before adopting it, since a more confident
classifier may also raise more false alerts there.

### Full benchmark (eval/gain_test.py, Colab, 2026-11-08): the gain does not help
Same final system, thresholds and gate settings; all 272 test streams (480 targets, 96 sirens), each stream
multiplied by the gain and clipped.

| Budget | Gain | Recall | Sirens (of 96) | FA/h events | FA/h idle | Classifier calls | Clipped | vs. +0 dB (all events; sirens) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | +0 dB | 44.2 % | 20 | 1.75 | 0.00 | 12 758 | 0 % | |
| 0.1 | +6 dB | 43.8 % | 19 | 2.00 | 0.00 | 12 758 | 0.006 % | +8/-10, p = 0.81; +0/-1 |
| 0.1 | +12 dB | 43.8 % | 19 | 1.75 | 0.00 | 12 752 | 0.068 % | +10/-12, p = 0.83; +1/-2 |
| 1 | +0 dB | 61.7 % | 51 | 10.27 | 0.50 | 19 159 | 0 % | |
| 1 | +6 dB | 62.1 % | 51 | 10.02 | 1.50 | 19 160 | 0.006 % | +7/-5, p = 0.77; +1/-1 |
| 1 | +12 dB | 61.5 % | 48 | 12.02 | 3.26 | 19 158 | 0.068 % | +9/-10, p = 1; +1/-4, p = 0.38 |

The 6 to 10 sirens of the MATLAB test do not hold: on 96 sirens the gain changes nothing at the strict budget and
slightly hurts at the relaxed one, where +12 dB also raises idle false alarms from 0.5 to 3.3 per hour. The gate is
unaffected (same classifier calls). The head was trained on audio at the benchmark's levels, so amplifying moves its
input away from what it learnt. **Recommendation: no gain.** The MATLAB result was small-sample noise, or specific
to the demo's synthetic pink-noise rooms.
