"""Step 2b: stress test of the alarm detector outside the conditions it was designed for.

The alarm detector and the test alarms both follow ISO 8201, so the main result could be
optimistic. Here alarms are varied beyond the specification (pitch outside the Goertzel probes,
timing +-30-50 %, heavy reverberation, the T-4 carbon-monoxide pattern), and look-alike sounds
that should NOT raise a fire alarm are added (a continuous 1 Hz beeper, a one-off triple beep
such as a microwave). Each variant: 20 seeds, 12 s of a test-environment background at -40 dBFS
with the sound from 2 s at 5 dB SNR. Reported: share of clips where each detector raises a fire
alarm: the classifier (F0 rule), the T-3 detector (loose / strict) and a "confirmed" rule that
needs two T-3 cycles.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import yamnet  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402

DUR, ONSET, SNR, SEEDS = 12.0, 2.0, 5.0, 20
VARIANTS = {  # name: (expected fire alarm?, synthesis parameters)
    "T-3 piezo 3.1 kHz (in spec)": (True, dict(f0=3100)),
    "T-3 520 Hz square (in spec)": (True, dict(f0=520, square=True)),
    "T-3 piezo 2.6 kHz": (True, dict(f0=2600)),
    "T-3 piezo 4.0 kHz": (True, dict(f0=4000)),
    "T-3 timing x0.7": (True, dict(f0=3100, scale=0.7)),
    "T-3 timing x1.5": (True, dict(f0=3100, scale=1.5)),
    "T-3 reverb RT60 1.5 s": (True, dict(f0=3100, rt60=1.5)),
    "T-4 CO alarm 3.1 kHz": (False, dict(f0=3100, on=0.1, off=0.1, n_on=4, pause=5.0)),
    "1 Hz beeper 3.1 kHz": (False, dict(f0=3100, n_on=10, pause=0.5, cycles=1)),
    "triple beep 2.8 kHz (once)": (False, dict(f0=2800, on=0.3, off=0.3, cycles=1)),
}


def alarm(seed, f0, square=False, on=0.5, off=0.5, n_on=3, pause=1.5, cycles=3, scale=1.0, rt60=None):
    rng = np.random.default_rng(seed)
    f0 = f0 * rng.uniform(0.98, 1.02)
    on, off, pause = on * scale, off * scale, pause * scale
    t = np.arange(int((DUR - ONSET) * C.SR)) / C.SR
    phase = 2 * np.pi * f0 * t
    tone = (sum(np.sin(k * phase) / k for k in range(1, 16, 2) if k * f0 < 7500) if square
            else np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase))
    env = np.zeros_like(t)
    start = 0.0
    for _ in range(cycles):
        for i in range(n_on):
            env[int(start * C.SR):int((start + on) * C.SR)] = 1
            start += on + (off if i < n_on - 1 else pause)
    dry = tone * np.convolve(env, np.ones(80) / 80, mode="same")
    n = int((rt60 or rng.uniform(0.3, 0.8)) * C.SR)
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    rir[0] = 8.0
    return signal.fftconvolve(dry, rir)[:len(dry)].astype(np.float32)


def scene(name, seed):
    """12 s of a test-environment background at -40 dBFS with variant `name` from 2 s at 5 dB SNR."""
    _, params = VARIANTS[name]
    rng = np.random.default_rng(seed + 1000)
    env = C.TEST_ENVS[seed % len(C.TEST_ENVS)]
    bg = streams.load(os.path.join(C.DATA, "demand", env, "ch01.wav"))
    i = int(rng.integers(0, len(bg) - int(DUR * C.SR)))
    bg = bg[i:i + int(DUR * C.SR)]
    bg = bg * 10 ** (-40 / 20) / (np.sqrt(np.mean(bg ** 2)) + 1e-9)
    a = alarm(seed, **params)
    _, _, rms = streams.active(a)
    x = bg.copy()
    x[int(ONSET * C.SR):] += a / rms * 10 ** (-40 / 20) * 10 ** (SNR / 20)
    return x.astype(np.float32)


def job(arg):
    name, seed = arg
    expected, params = VARIANTS[name]
    rng = np.random.default_rng(seed + 1000)
    env = C.TEST_ENVS[seed % len(C.TEST_ENVS)]
    bg = streams.load(os.path.join(C.DATA, "demand", env, "ch01.wav"))
    i = int(rng.integers(0, len(bg) - int(DUR * C.SR)))
    bg = bg[i:i + int(DUR * C.SR)]
    bg = bg * 10 ** (-40 / 20) / (np.sqrt(np.mean(bg ** 2)) + 1e-9)
    a = alarm(seed, **params)
    _, _, rms = streams.active(a)
    x = bg.copy()
    x[int(ONSET * C.SR):] += a / rms * 10 ** (-40 / 20) * 10 ** (SNR / 20)
    on = frontend.tone_track(x)
    loose, strict = frontend.t3_times(on), frontend.t3_times(on, strict=True)
    confirmed = [t for t in strict if any(3.0 <= t - u <= 5.0 for u in strict)]
    v4 = frontend.alarm_times(x, cfg=frontend.chosen_alarm())
    scores, emb = yamnet.window_outputs(x)
    cat = yamnet.category_scores(scores)
    yam = any(c == "FIRE_ALARM" for _, c in yamnet.detect(cat, siren_veto=True))
    from classifier import head
    hd = head.available() and any(c == "FIRE_ALARM" for _, c in yamnet.detect(
        head.category_scores(emb, scores), thresholds=head.load()["thresholds"], siren_veto=True))
    from eval.evaluate import agreed
    v4_agree = agreed([(t, "FIRE_ALARM") for t in v4], cat, None, C.ORDER.index("FIRE_ALARM"), C.ALARM_AGREE)
    return name, dict(yamnet=yam, head=bool(hd), t3_loose=bool(loose), t3_strict=bool(strict), t3_confirmed=bool(confirmed),
                      dsp_v4=bool(v4), dsp_v4_agree=bool(v4_agree))


def main():
    jobs = [(n, s) for n in VARIANTS for s in range(SEEDS)]
    with Pool(C.WORKERS) as pool:
        res = pool.map(job, jobs)
    out = {}
    for n in VARIANTS:
        rs = [r for k, r in res if k == n]
        out[n] = dict(expected_fire=VARIANTS[n][0], **{d: float(np.mean([r[d] for r in rs])) for d in rs[0]})
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "stress.json"), "w"), indent=1)
    print(f"\n  {'variant':30s} {'should fire':>11s} {'YAMNet':>7s} {'head':>6s} {'T-3':>6s} {'strict':>7s} {'2 cycles':>8s} {'v4':>6s} {'v4+agree':>8s}")
    for n, r in out.items():
        print(f"  {n:30s} {'yes' if r['expected_fire'] else 'no':>11s} {r['yamnet']:7.2f} {r['head']:6.2f} {r['t3_loose']:6.2f} "
              f"{r['t3_strict']:7.2f} {r['t3_confirmed']:8.2f} {r['dsp_v4']:6.2f} {r['dsp_v4_agree']:8.2f}")


if __name__ == "__main__":
    main()
