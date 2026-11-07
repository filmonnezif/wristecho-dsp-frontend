"""Test streams: 5-minute DEMAND recordings with sparse target events and distractors mixed in.

Each environment x microphone gives two streams:
  "events"  6 targets (2 fire alarms, 2 sirens, 2 door sounds; one per 50 s slot) at an SNR from
            SNRS_DB, plus 6 distractors (everyday sounds, half "hard") placed so they never
            overlap a target.
  "idle"    the same background alone: what the watch hears most of the day.
The background level varies per stream (BG_DBFS), as room loudness and mic gain do. SNR = event
power over its active frames / background power. Targets come from the stream's split only, so
test streams contain no clip, alarm seed or environment used for tuning.
"""
import json
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal

import config as C


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != C.SR:
        g = gcd(sr, C.SR)
        x = signal.resample_poly(x, C.SR // g, sr // g).astype(np.float32)
    return x


def active(x, below_db=35):
    """(first active sample, end sample, active RMS) over 20 ms frames within 35 dB of the peak."""
    f = 320
    n = max(1, len(x) // f)
    rms = np.sqrt(np.mean(np.pad(x, (0, max(0, n * f - len(x))))[:n * f].reshape(n, f) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    mask = db > db.max() - below_db
    idx = np.flatnonzero(mask)
    return idx[0] * f, (idx[-1] + 1) * f, float(np.sqrt(np.mean(rms[mask] ** 2)))


def synthetic_alarm(seed):
    """ISO 8201 T-3 smoke alarm, two cycles (8 s): 3 x (0.5 s on, 0.5 s off) + 1.5 s off, with
    jittered timing and pitch, slight FM, 5 ms ramps and a synthetic room impulse response.
    Even seeds: piezo (2.9-3.5 kHz + 2nd/3rd harmonics). Odd seeds: 520 Hz square wave."""
    rng = np.random.default_rng(seed)
    square = seed % 2 == 1
    f0 = 520 * rng.uniform(0.97, 1.03) if square else rng.uniform(2900, 3500)
    t = np.arange(int(8.0 * C.SR)) / C.SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.003 * np.sin(2 * np.pi * rng.uniform(3, 8) * t))) / C.SR
    if square:  # band-limited: odd harmonics below 7.5 kHz, so nothing aliases at fs = 16 kHz
        tone = sum(np.sin(k * phase) / k for k in range(1, 16, 2) if k * f0 < 7500)
    else:
        tone = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    env = np.zeros_like(t)
    on, off = 0.5 * rng.uniform(0.9, 1.1), 0.5 * rng.uniform(0.9, 1.1)
    start = 0.0
    for _ in range(2):
        for _ in range(3):
            env[int(start * C.SR):int((start + on) * C.SR)] = 1
            start += on + off
        start += 1.5 - off
    dry = tone * np.convolve(env, np.ones(80) / 80, mode="same")
    n = int(rng.uniform(0.3, 0.8) * C.SR)  # RT60
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    rir[0] = 8.0
    wet = signal.fftconvolve(dry, rir)[:len(dry)]
    return (wet / np.max(np.abs(wet))).astype(np.float32)


def clip_path(c):
    if c["src"] == "esc":
        return os.path.join(C.DATA, "ESC-50-master", "audio", c["id"])
    return os.path.join(C.DATA, "fsd50k", "clips", c["dir"], c["id"] + ".wav")


def specs():
    """All stream specs (deterministic from config.SEED)."""
    clips = json.load(open(os.path.join(C.ROOT, "data", "cliplist.json")))
    rng = np.random.default_rng(C.SEED)
    out = []
    for split, envs in (("tune", C.TUNE_ENVS), ("test", C.TEST_ENVS)):
        pool = {k: [c for c in clips if c["split"] == split and c["role"] == "target" and c["category"] == k]
                for k in ("SIREN", "DOOR")}
        dis = [c for c in clips if c["split"] == split and c["role"] == "distractor"]
        alarms = list(C.ALARM_SEEDS[split])
        for env in envs:
            for ch in C.CHANNELS:
                cats = ["FIRE_ALARM", "SIREN", "DOOR"] * 2
                rng.shuffle(cats)
                slot = C.STREAM_S / C.TARGETS_PER_STREAM
                events = []
                for i, cat in enumerate(cats):
                    if cat == "FIRE_ALARM":
                        seed = int(rng.choice(alarms))
                        src = dict(kind="alarm", seed=seed, subtype="piezo" if seed % 2 == 0 else "520hz")
                    else:
                        c = pool[cat][rng.integers(len(pool[cat]))]
                        src = dict(kind="clip", clip=c, subtype=c["subtype"])
                    events.append(dict(role="target", category=cat, source=src,
                                       onset=float(i * slot + rng.uniform(2, slot - C.MAX_EVENT_S - 4)),
                                       snr=float(rng.choice(C.SNRS_DB))))
                for _ in range(C.DISTRACTORS_PER_STREAM):
                    c = dis[rng.integers(len(dis))]
                    events.append(dict(role="distractor", category=None, source=dict(kind="clip", clip=c, subtype=c["subtype"]),
                                       onset=float(rng.uniform(0, C.STREAM_S - C.MAX_EVENT_S)),  # placed in render()
                                       snr=float(rng.uniform(*C.DISTRACTOR_SNR_DB))))
                bg_dbfs = float(rng.uniform(*C.BG_DBFS))
                for kind in ("events", "idle"):
                    out.append(dict(id=f"{split}-{env}-ch{ch:02d}-{kind}", split=split, env=env, ch=ch, kind=kind,
                                    bg_dbfs=bg_dbfs, events=events if kind == "events" else [],
                                    place_seed=int(rng.integers(1 << 30))))
    return out


def event_audio(src):
    x = synthetic_alarm(src["seed"]) if src["kind"] == "alarm" else load(clip_path(src["clip"]))
    a, b, rms = active(x)
    x = x[a:b][:int(C.MAX_EVENT_S * C.SR)]
    ramp = min(160, len(x) // 2)
    if ramp:
        x[:ramp] *= np.linspace(0, 1, ramp)
        x[-ramp:] *= np.linspace(1, 0, ramp)
    return x / rms


def render(spec):
    """(audio, labelled events with final onset/end). Distractors are re-placed so they never
    overlap a target (+-1 s)."""
    bg = load(os.path.join(C.DATA, "demand", spec["env"], f"ch{spec['ch']:02d}.wav"))[:C.STREAM_S * C.SR]
    bg_rms = float(np.sqrt(np.mean(bg ** 2))) + 1e-9
    g_bg = 10 ** (spec["bg_dbfs"] / 20) / bg_rms
    x = bg * g_bg
    rng = np.random.default_rng(spec["place_seed"])
    placed, busy = [], []
    for ev in sorted(spec["events"], key=lambda e: e["role"] != "target"):  # targets first
        a = event_audio(ev["source"])
        dur = len(a) / C.SR
        onset = ev["onset"]
        if ev["role"] == "distractor":
            for _ in range(200):
                onset = float(rng.uniform(0, C.STREAM_S - dur))
                if all(onset + dur < s - 1 or onset > e + 1 for s, e in busy):
                    break
            else:
                continue
        i = int(onset * C.SR)
        a = a[:len(x) - i]
        x[i:i + len(a)] += a * 10 ** (spec["bg_dbfs"] / 20) * 10 ** (ev["snr"] / 20)
        if ev["role"] == "target":
            busy.append((onset, onset + dur))
        placed.append(dict(role=ev["role"], category=ev["category"], subtype=ev["source"]["subtype"],
                           onset=onset, end=onset + len(a) / C.SR, snr=ev["snr"]))
    return np.clip(x, -1, 1).astype(np.float32), placed
