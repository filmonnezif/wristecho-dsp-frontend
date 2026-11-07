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


def novelty(bands_db, smooth_s, rank, rise_db_s=None):
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
    rise = (C.FLOOR_RISE_DB_S if rise_db_s is None else rise_db_s) * C.FRAME / C.SR
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
    """Per-window gate statistic for one variant (dict(smooth_s, rank, cfar[, slow_rise, slow_offset])).

    Two-floor variant: a second floor that rises only slow_rise dB/s remembers how quiet the room
    was, so a sound that swells slower than the fast floor rises (a siren fading in) still stands
    out above it. The statistic is max(fast rise, slow rise - slow_offset)."""
    w = per_window(novelty(bands_db, variant["smooth_s"], variant["rank"]), n_windows)
    if variant.get("slow_rise"):
        slow = per_window(novelty(bands_db, variant["smooth_s"], variant["rank"], variant["slow_rise"]), n_windows)
        w = np.maximum(w, slow - variant["slow_offset"])
    return os_cfar(w) if variant["cfar"] else w


# --- Knock detector -----------------------------------------------------------------------------

def onset_strength(bands_db):
    """Per frame: mean over bands of the half-wave rectified rise of the band level since the
    previous frame (spectral flux in dB). A knock is a short broadband impulse: a large jump in
    every band at once."""
    d = np.diff(bands_db.astype(np.float32), axis=0, prepend=bands_db[:1])
    return np.maximum(d, 0).mean(axis=1)


def knock_times(flux, kappa, min_hits):
    """Times (s) at which a knock pattern completes: min_hits impulses (local flux maxima above
    kappa dB, at least 60 ms apart) with inter-onset intervals of 0.12-0.6 s, and for three or more
    a regular rhythm (each interval within a factor 1.6 of the previous)."""
    fs = C.FRAME / C.SR
    pad = np.r_[np.zeros(3), flux, np.zeros(3)]
    local = np.lib.stride_tricks.sliding_window_view(pad, 7).max(axis=1)
    peaks = np.flatnonzero((flux >= local) & (flux > kappa)) * fs
    hits = []
    for i in range(len(peaks) - min_hits + 1):
        q = peaks[i:i + min_hits]
        ioi = np.diff(q)
        if ((ioi >= 0.12) & (ioi <= 0.6)).all() and (len(ioi) < 2 or (np.maximum(ioi[1:], ioi[:-1]) /
                                                                     np.minimum(ioi[1:], ioi[:-1]) <= 1.6).all()):
            hits.append(float(q[-1]))
    return hits


def gate_features(x):
    """Per frame: level and the v1 novelty (largest band rise, 60 ms smoothing). Kept for tests."""
    level, bands = band_power(x)
    return level, novelty(bands, C.BAND_SMOOTH_S, 1)


def knock_ops_per_window():
    """Knock detector per hop: per frame 16 differences + 16 max + mean, 7-point max for peaks."""
    return int((3 * C.N_BANDS + 8) * C.HOP // C.FRAME)


def gate_ops_per_window():
    """Multiply-accumulates of the gate per classifier hop (0.5 s = 25 frames), counted per stage:
    biquad HPF 5/sample, level 1/sample, Hann 1/sample, real FFT ~N log2 N, |X|^2 2/bin,
    band sums 1/bin, then per band: smoothing 2, log 1, floor + novelty 3, sort 4; per window the
    CFAR median of 20 values (~60 comparisons)."""
    per_frame = (5 + 1 + 1) * C.FRAME + C.NFFT * np.log2(C.NFFT) + 3 * (C.NFFT // 2 + 1) + 10 * C.N_BANDS
    return int(per_frame * C.HOP // C.FRAME + 60)


def alarm_ops_per_window(cfg=None):
    """Goertzel bank per hop: Hann + frame energy 2/sample, 1 MAC/sample per probe, 3 per readout."""
    cfg = cfg or C.ALARM_V3
    probes = len(np.arange(cfg["lo"], cfg["hi"] + 1, C.PROBE_STEP)) + len(C.SQUARE_PROBES)
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


def probe_energies(x, freqs):
    """Goertzel energies per 20 ms frame for the piezo probes `freqs` (frames x probes), the summed
    square-wave probes (520 Hz + odd harmonics), the frame energy and a loudness mask."""
    fr = _frames(x) * np.hanning(C.FRAME)
    total = np.sum(fr ** 2, axis=1) * C.FRAME / 2 + 1e-12
    e = np.stack([goertzel(fr, f) for f in freqs], axis=1)
    sq = np.sum([goertzel(fr, f) for f in C.SQUARE_PROBES], axis=0)
    return e, sq, total, 10 * np.log10(total / C.FRAME) > -75


def _envelope_ok(e):
    """Room reverb keeps the gaps tonal: a frame is "on" only within 8 dB of its 1.5 s running peak."""
    db = 10 * np.log10(e + 1e-12)
    pad = np.r_[np.full(74, -300.0), db]
    return db > np.lib.stride_tricks.sliding_window_view(pad, 75).max(axis=1) - 8


def tone_on(e, sq, total, loud, freqs, lo, hi, ratio):
    """Per frame: True when one piezo probe in [lo, hi] Hz holds more than `ratio` of the frame
    energy (or the square-wave probes more than 35 %), i.e. the frame is a narrowband tone."""
    sel = (freqs >= lo) & (freqs <= hi)
    piezo_e = e[:, sel].max(axis=1)
    return (((piezo_e / total > ratio) & _envelope_ok(piezo_e)) | ((sq / total > 0.35) & _envelope_ok(sq))) & loud


def tone_track(x, cfg=None):
    """Per 20 ms frame: tonal alarm frames for detector configuration cfg (default: the first
    design, 2.8-3.6 kHz probes, 25 % of the frame energy)."""
    cfg = cfg or C.ALARM_V3
    freqs = np.arange(cfg["lo"], cfg["hi"] + 1, C.PROBE_STEP)
    return tone_on(*probe_energies(x, freqs), freqs, cfg["lo"], cfg["hi"], cfg["ratio"])


def _pulses(on):
    fs = C.FRAME / C.SR
    on = ndimage.binary_closing(np.r_[False, on, False], np.ones(6))[1:-1]  # bridge 100 ms dropouts
    on = signal.medfilt(on.astype(float), 3) > 0.5  # drop blips
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(edges[i] * fs, edges[i + 1] * fs) for i in range(0, len(edges), 2)], len(on) * fs


def periodic_times(on, min_pulses):
    """Times (s) at which `min_pulses` regular tonal beeps have been heard: each 0.05-1.0 s long,
    gaps 0.05-1.5 s, and a steady rhythm (coefficient of variation of the beep period and length
    below 0.25). Covers alarms that beep continuously instead of in the T-3 rhythm."""
    pulses, _ = _pulses(on)
    hits = []
    for i in range(len(pulses) - min_pulses + 1):
        p = pulses[i:i + min_pulses]
        dur = np.array([b - a for a, b in p])
        gaps = np.array([p[k + 1][0] - p[k][1] for k in range(min_pulses - 1)])
        period = np.diff([a for a, _ in p])
        if ((0.05 <= dur) & (dur <= 1.0)).all() and ((0.05 <= gaps) & (gaps <= 1.5)).all() \
                and period.std() / period.mean() < 0.25 and dur.std() / dur.mean() < 0.25:
            hits.append(p[-1][1])
    return hits


def decide(on, rule):
    """Alarm times for a pattern rule: "t3", "t3_strict", "periodicN", or alternatives joined by |."""
    out = []
    for r in rule.split("|"):
        if r == "t3":
            out += t3_times(on)
        elif r == "t3_strict":
            out += t3_times(on, strict=True)
        elif r.startswith("periodic"):
            out += periodic_times(on, int(r[len("periodic"):]))
    return sorted(out)


def chosen_alarm():
    """The detector configuration selected by eval/alarm_design.py, or the first design."""
    import json
    import os
    p = os.path.join(C.RESULTS, "alarm_design.json")
    return json.load(open(p))["chosen"] if os.path.exists(p) else C.ALARM_V3


def t3_times(on, strict=False):
    """Times (s) at which a T-3 pattern completes: 3 pulses of 0.3-0.8 s with gaps of 0.3-0.8 s
    (ISO 8201: 0.5 s on, 0.5 s off, three times, then 1.5 s off). strict: also require the
    1.5 s pause (no new pulse for 1.0 s after the third), which rejects continuous 1 Hz beepers
    at the cost of ~1 s more latency."""
    pulses, length = _pulses(on)
    hits = []
    for i in range(len(pulses) - 2):
        p = pulses[i:i + 3]
        if all(0.3 <= b - a <= 0.8 for a, b in p) and all(0.3 <= p[k + 1][0] - p[k][1] <= 0.8 for k in range(2)):
            if not strict:
                hits.append(p[2][1])
                continue
            nxt = pulses[i + 3][0] if i + 3 < len(pulses) else np.inf
            prev_gap = p[0][0] - pulses[i - 1][1] if i > 0 else np.inf
            if nxt - p[2][1] >= 1.0 and prev_gap >= 1.0 and length >= p[2][1] + 1.0:
                hits.append(p[2][1] + 1.0)  # decided once the pause has lasted 1 s
    return hits


def alarm_times(x, strict=False, cfg=None):
    """Fire-alarm times. cfg=None: the first design (T-3, loose or strict); otherwise cfg's rule."""
    if cfg is None:
        return t3_times(tone_track(x), strict)
    return decide(tone_track(x, cfg), cfg["rule"])
