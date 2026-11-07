"""The DSP front end: everything that runs on every sample, before (or instead of) the classifier.

Gate path   x -> 2nd-order Butterworth HPF (100 Hz) -> 20 ms frames -> |FFT|^2 in 16 log bands
            -> one-pole smoothing -> per-band noise floor (minimum tracking) -> novelty = rank-th
            largest band rise [dB] -> optional ordered-statistic CFAR (median of the last 10 s)
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

def band_power(x):
    """Per 20 ms frame, after the HPF: level (dBFS) and power in N_BANDS log-spaced bands (dB).
    This is the only part of the gate that touches samples; everything after works on 16 numbers
    per frame, so gate variants can be compared from cached band powers."""
    y = signal.sosfilt(hpf_sos(), x).astype(np.float32)
    fr = _frames(y)
    level = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(fr * np.hanning(C.FRAME), n=C.NFFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(C.NFFT, 1 / C.SR)
    edges = band_edges()
    bands = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    return level.astype(np.float32), (10 * np.log10(bands + 1e-10)).astype(np.float32)


def novelty(bands_db, smooth_s, rank):
    """Per frame: the rank-th largest rise of a band above its own noise floor (dB).

    1. Smooth each band's power with a one-pole low-pass, y[t] = (1-a) x[t] + a y[t-1],
       a = exp(-T/smooth_s): lowers the frame-to-frame variance of the few-bin low bands.
    2. Noise floor per band (minimum tracking): F[t] = min(E[t], F[t-1] + rise). It follows the band
       down at once and climbs at most FLOOR_RISE_DB_S, so a steady sound is absorbed in seconds.
    3. rank = 1 takes the largest band rise; rank = 2 needs two bands to rise, which ignores a single
       fluctuating band (most new sounds are broadband or have harmonics).
    """
    a = np.exp(-C.FRAME / C.SR / smooth_s)
    p = signal.lfilter([1 - a], [1, -a], 10 ** (bands_db.astype(np.float64) / 10), axis=0)
    e = 10 * np.log10(p + 1e-10)
    rise = C.FLOOR_RISE_DB_S * C.FRAME / C.SR
    floor = np.empty_like(e)
    f = e[:50].min(axis=0)  # initial floor: quietest of the first second
    for t in range(len(e)):
        f = np.minimum(e[t], f + rise)
        floor[t] = f
    return np.sort(e - floor, axis=1)[:, -rank].astype(np.float32)


def os_cfar(nov_w, ref_s=C.CFAR_REF_S):
    """Ordered-statistic CFAR on the per-window novelty: subtract the median of the previous
    ref_s seconds, so the threshold rides on the room's own typical novelty (busy cafe vs quiet
    bedroom) instead of a fixed dB value. Causal: only past windows are used."""
    n = int(ref_s / (C.HOP / C.SR))
    out = np.empty_like(nov_w)
    for k in range(len(nov_w)):
        ref = nov_w[max(0, k - n):k]
        out[k] = nov_w[k] - (np.median(ref) if len(ref) else nov_w[k])
    return out


def gate_novelty(bands_db, n_windows, variant):
    """Per-window gate statistic for one variant (dict(smooth_s, rank, cfar))."""
    w = per_window(novelty(bands_db, variant["smooth_s"], variant["rank"]), n_windows)
    return os_cfar(w) if variant["cfar"] else w


def gate_features(x):
    """Per frame: level and the v1 novelty (largest band rise, 60 ms smoothing). Kept for tests."""
    level, bands = band_power(x)
    return level, novelty(bands, C.BAND_SMOOTH_S, 1)


def gate_ops_per_window():
    """Multiply-accumulates of the gate per classifier hop (0.5 s = 25 frames), counted per stage:
    biquad HPF 5/sample, level 1/sample, Hann 1/sample, real FFT ~N log2 N, |X|^2 2/bin,
    band sums 1/bin, then per band: smoothing 2, log 1, floor + novelty 3, sort 4; per window the
    CFAR median of 20 values (~60 comparisons)."""
    per_frame = (5 + 1 + 1) * C.FRAME + C.NFFT * np.log2(C.NFFT) + 3 * (C.NFFT // 2 + 1) + 10 * C.N_BANDS
    return int(per_frame * C.HOP // C.FRAME + 60)


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


def t3_times(on, strict=False):
    """Times (s) at which a T-3 pattern completes: 3 pulses of 0.3-0.8 s with gaps of 0.3-0.8 s
    (ISO 8201: 0.5 s on, 0.5 s off, three times, then 1.5 s off). strict: also require the
    1.5 s pause (no new pulse for 1.0 s after the third), which rejects continuous 1 Hz beepers
    at the cost of ~1 s more latency."""
    fs = C.FRAME / C.SR
    on = ndimage.binary_closing(np.r_[False, on, False], np.ones(6))[1:-1]  # bridge 100 ms dropouts
    on = signal.medfilt(on.astype(float), 3) > 0.5  # drop blips
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    pulses = [(edges[i] * fs, edges[i + 1] * fs) for i in range(0, len(edges), 2)]
    hits = []
    for i in range(len(pulses) - 2):
        p = pulses[i:i + 3]
        if all(0.3 <= b - a <= 0.8 for a, b in p) and all(0.3 <= p[k + 1][0] - p[k][1] <= 0.8 for k in range(2)):
            if not strict:
                hits.append(p[2][1])
                continue
            nxt = pulses[i + 3][0] if i + 3 < len(pulses) else np.inf
            prev_gap = p[0][0] - pulses[i - 1][1] if i > 0 else np.inf
            if nxt - p[2][1] >= 1.0 and prev_gap >= 1.0 and len(on) * fs >= p[2][1] + 1.0:
                hits.append(p[2][1] + 1.0)  # decided once the pause has lasted 1 s
    return hits


def alarm_times(x, strict=False):
    return t3_times(tone_track(x), strict)
