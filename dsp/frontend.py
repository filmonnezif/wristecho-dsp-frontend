"""The DSP front end: everything that runs on every sample, before (or instead of) the classifier.

Gate path   x -> 2nd-order Butterworth HPF (100 Hz) -> 20 ms frames -> |FFT|^2 in 16 log bands
            -> per-band noise floor (minimum tracking) -> novelty = max_b (E_b - floor_b) [dB]
Alarm path  x -> Goertzel resonator bank (piezo 2.8-3.6 kHz, 520 Hz square + harmonics)
            -> tonal on/off track -> T-3 temporal pattern matcher (ISO 8201)
"""
import numpy as np
from scipy import ndimage, signal

import config as C

# --- Filters ------------------------------------------------------------------------------------

def hpf_sos():
    """Spec: Butterworth high-pass, fc = 100 Hz (-3 dB), order 2 (-40 dB/decade), fs = 16 kHz.
    Removes DC offset, handling rumble and wind below the lowest target component (520 Hz)."""
    return signal.butter(C.HPF_ORDER, C.HPF_HZ, btype="highpass", fs=C.SR, output="sos")


def band_edges():
    return np.geomspace(C.BAND_LO, C.BAND_HI, C.N_BANDS + 1)


def _frames(x):
    n = len(x) // C.FRAME
    return x[:n * C.FRAME].reshape(n, C.FRAME)


# --- Gate features ------------------------------------------------------------------------------

def gate_features(x):
    """Per 20 ms frame: level (dBFS, after HPF) and novelty (dB above the adaptive noise floor).

    Band powers are first smoothed over ~60 ms (one-pole IIR). Noise floor per band: F[t] = E[t] if E[t] < F[t-1] else F[t-1] + rise, i.e. it follows the
    band down at once and climbs at most FLOOR_RISE_DB_S. A new sound lifts E far above F (high
    novelty); a steady sound is absorbed into F within seconds, so the gate goes back to sleep.
    """
    y = signal.sosfilt(hpf_sos(), x).astype(np.float32)
    fr = _frames(y)
    level = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(fr * np.hanning(C.FRAME), n=C.NFFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(C.NFFT, 1 / C.SR)
    edges = band_edges()
    bands = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    a = np.exp(-C.FRAME / C.SR / C.BAND_SMOOTH_S)  # one-pole low-pass, y[t] = (1-a) x[t] + a y[t-1]
    bands = signal.lfilter([1 - a], [1, -a], bands, axis=0)
    e = 10 * np.log10(bands + 1e-10)
    rise = C.FLOOR_RISE_DB_S * C.FRAME / C.SR
    floor = np.empty_like(e)
    f = e[:50].min(axis=0)  # initial floor: quietest of the first second
    for t in range(len(e)):
        f = np.minimum(e[t], f + rise)
        floor[t] = f
    novelty = (e - floor).max(axis=1)
    return level.astype(np.float32), novelty.astype(np.float32)


def gate_ops_per_window():
    """Multiply-accumulates of the gate per classifier hop (0.5 s = 25 frames), counted per stage:
    biquad HPF 5/sample, level 1/sample, Hann 1/sample, real FFT ~N log2 N, |X|^2 2/bin,
    band sums 1/bin, then per band: smoothing 2, log 1, floor + novelty 3."""
    per_frame = (5 + 1 + 1) * C.FRAME + C.NFFT * np.log2(C.NFFT) + 3 * (C.NFFT // 2 + 1) + 6 * C.N_BANDS
    return int(per_frame * C.HOP // C.FRAME)


def alarm_ops_per_window():
    """Goertzel bank per hop: Hann + frame energy 2/sample, 1 MAC/sample per probe, 3 per readout."""
    probes = len(C.PIEZO_PROBES) + len(C.SQUARE_PROBES)
    per_frame = 2 * C.FRAME + probes * (C.FRAME + 3)
    return int(per_frame * C.HOP // C.FRAME)


def per_window(frame_values, n_windows):
    """Max of a per-frame feature over the 0.5 s of new audio before each classifier window ends.
    Window k ends at sample WIN + k*HOP; the gate decides from the hop that just arrived."""
    fpw = C.HOP // C.FRAME
    out = np.full(n_windows, -np.inf, np.float32)
    for k in range(n_windows):
        end = (C.WIN + k * C.HOP) // C.FRAME
        out[k] = frame_values[max(0, end - fpw):end].max()
    return out


# --- Goertzel alarm detector --------------------------------------------------------------------

def goertzel(frames, freq):
    """Power of one DFT bin per frame via s[n] = x[n] + 2cos(w) s[n-1] - s[n-2]: a 2nd-order IIR
    resonator with poles at exp(+-jw) on the unit circle, read out once after N samples."""
    w = 2 * np.pi * freq / C.SR
    c = 2 * np.cos(w)
    s1 = np.zeros(len(frames))
    s2 = np.zeros(len(frames))
    for k in range(frames.shape[1]):
        s1, s2 = frames[:, k] + c * s1 - s2, s1
    return s1 * s1 + s2 * s2 - c * s1 * s2


def tone_track(x):
    """Per 20 ms frame: True when one alarm probe holds a large share of the frame energy."""
    fr = _frames(x) * np.hanning(C.FRAME)
    total = np.sum(fr ** 2, axis=1) * C.FRAME / 2 + 1e-12
    piezo_e = np.max([goertzel(fr, f) for f in C.PIEZO_PROBES], axis=0)
    square_e = np.sum([goertzel(fr, f) for f in C.SQUARE_PROBES], axis=0)
    loud = 10 * np.log10(total / C.FRAME) > -75

    def envelope_ok(e):  # room reverb keeps gaps tonal: "on" only within 8 dB of the 1.5 s peak
        db = 10 * np.log10(e + 1e-12)
        pad = np.r_[np.full(74, -300.0), db]
        return db > np.lib.stride_tricks.sliding_window_view(pad, 75).max(axis=1) - 8

    piezo = (piezo_e / total > 0.25) & envelope_ok(piezo_e)
    square = (square_e / total > 0.35) & envelope_ok(square_e)
    return (piezo | square) & loud


def t3_times(on):
    """Times (s) at which a T-3 pattern completes: 3 pulses of 0.3-0.8 s with gaps of 0.3-0.8 s
    (ISO 8201: 0.5 s on, 0.5 s off, three times, then 1.5 s off)."""
    fs = C.FRAME / C.SR
    on = ndimage.binary_closing(np.r_[False, on, False], np.ones(6))[1:-1]  # bridge 100 ms dropouts
    on = signal.medfilt(on.astype(float), 3) > 0.5  # drop blips
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    pulses = [(edges[i] * fs, edges[i + 1] * fs) for i in range(0, len(edges), 2)]
    hits = []
    for i in range(len(pulses) - 2):
        p = pulses[i:i + 3]
        if all(0.3 <= b - a <= 0.8 for a, b in p) and all(0.3 <= p[k + 1][0] - p[k][1] <= 0.8 for k in range(2)):
            hits.append(p[2][1])
    return hits


def alarm_times(x):
    return t3_times(tone_track(x))
