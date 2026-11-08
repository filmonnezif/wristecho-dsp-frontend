"""Controlled test cases for the DSP blocks (python -m pytest tests)."""
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier.yamnet import detect  # noqa: E402
from data.streams import synthetic_alarm  # noqa: E402
from dsp import frontend  # noqa: E402


def test_goertzel_equals_dft_bin():
    rng = np.random.default_rng(0)
    fr = rng.standard_normal((5, C.FRAME))
    k = 62  # bin 62 of N = 320 at 16 kHz = 3100 Hz
    dft = np.abs(np.fft.fft(fr, axis=1)[:, k]) ** 2
    assert np.allclose(frontend.goertzel(fr, k * C.SR / C.FRAME), dft, rtol=1e-6)


def test_hpf_meets_spec():
    w, h = signal.sosfreqz(frontend.hpf_sos(), worN=[50, 100, 520, 3000], fs=C.SR)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[1] + 3.01) < 0.1  # -3 dB at the cutoff
    assert db[0] < -11  # 2nd order: ~-12 dB one octave below
    assert db[2] > -0.2 and db[3] > -0.01  # alarm tones pass


def test_t3_detected_on_both_alarm_types():
    for seed in (0, 1):  # piezo, 520 Hz square
        x = 0.1 * synthetic_alarm(seed)
        assert len(frontend.alarm_times(x)) >= 1


def test_t3_silent_on_noise_and_steady_tone():
    rng = np.random.default_rng(1)
    noise = 0.01 * rng.standard_normal(8 * C.SR).astype(np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 3100 * np.arange(8 * C.SR) / C.SR).astype(np.float32)
    assert frontend.alarm_times(noise) == [] and frontend.alarm_times(tone) == []


def test_novelty_wakes_on_onset_and_sleeps_on_steady():
    rng = np.random.default_rng(2)
    x = 0.003 * rng.standard_normal(20 * C.SR).astype(np.float32)
    x[5 * C.SR:15 * C.SR] += 0.1 * np.sin(2 * np.pi * 1000 * np.arange(10 * C.SR) / C.SR)
    _, nov = frontend.gate_features(x)
    t = np.arange(len(nov)) * C.FRAME / C.SR
    onset = nov[(t > 5) & (t < 5.2)].max()
    assert onset > nov[(t > 2) & (t < 4.5)].max() + 25  # onset far above the background's novelty
    assert nov[(t > 14) & (t < 14.9)].max() < onset - 20  # the floor climbs under a steady tone


def test_detect_rule_and_sleeping_windows():
    cat = np.zeros((40, len(C.ORDER), 2), np.float32)
    j = C.ORDER.index("SIREN")
    cat[10:12, j, 0] = 0.9  # two consecutive windows -> one siren
    assert [c for _, c in detect(cat)] == ["SIREN"]
    awake = np.ones(40, bool)
    awake[11] = False  # one skipped window: the run continues with the next classified one
    cat[12, j, 0] = 0.9
    assert [c for _, c in detect(cat, awake)] == ["SIREN"]
    awake[12:20] = False  # a long sleep (> HOLD_S) breaks the run
    assert detect(cat, awake) == []
    assert [c for _, c in detect(cat, awake, extra=[(3.0, "FIRE_ALARM")])] == ["FIRE_ALARM"]


def test_strict_t3_needs_the_pause():
    x = 0.1 * synthetic_alarm(0)
    assert len(frontend.alarm_times(x, strict=True)) >= 1
    t = np.arange(10 * C.SR) / C.SR  # continuous 1 Hz beeper at 3.1 kHz: no 1.5 s pause
    beeper = (0.1 * np.sin(2 * np.pi * 3100 * t) * ((t % 1.0) < 0.5)).astype(np.float32)
    assert frontend.alarm_times(beeper) and not frontend.alarm_times(beeper, strict=True)


def test_cfar_centres_background():
    rng = np.random.default_rng(3)
    w = rng.normal(12, 2, 200).astype(np.float32)
    w[150] = 40
    c = frontend.os_cfar(w)
    assert abs(np.median(c[40:])) < 1.5 and c[150] > 25


def test_two_floor_gate_catches_a_slow_swell():
    rng = np.random.default_rng(4)
    t = np.arange(30 * C.SR) / C.SR
    x = 0.003 * rng.standard_normal(len(t))
    amp = np.clip((t - 5) / 12, 0, 1) * 0.1  # a 1 kHz tone swelling over 12 s (~3.3 dB/s at the end)
    x = (x + amp * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    _, bands = frontend.band_power(x)
    n = (len(x) - C.WIN) // C.HOP + 1
    fast = frontend.gate_novelty(bands, n, C.GATE_VARIANTS["v1"])
    two = frontend.gate_novelty(bands, n, C.GATE_VARIANTS["two-floor+10"])
    late = (C.WIN + np.arange(n) * C.HOP) / C.SR > 15
    assert two[late].max() > fast[late].max() + 10


def test_knock_rhythm():
    rng = np.random.default_rng(5)
    x = 0.002 * rng.standard_normal(6 * C.SR).astype(np.float32)
    for t0 in (2.0, 2.3, 2.6):  # three knocks, 0.3 s apart
        i = int(t0 * C.SR)
        x[i:i + 400] += 0.3 * rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    _, bands = frontend.band_power(x)
    flux = frontend.onset_strength(bands)
    assert frontend.knock_times(flux, 8, 3)
    one = 0.002 * rng.standard_normal(6 * C.SR).astype(np.float32)
    one[32000:32400] += 0.3 * rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    assert not frontend.knock_times(frontend.onset_strength(frontend.band_power(one)[1]), 8, 2)

