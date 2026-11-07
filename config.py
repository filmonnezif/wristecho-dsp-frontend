"""Every parameter of the study in one place. Nothing else in the code hides a number that matters."""
import os

import numpy as np

# --- Paths --------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "datasets"))  # downloaded datasets
CACHE = os.environ.get("CACHE_DIR", os.path.join(ROOT, "cache"))  # per-stream features and scores
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures", "out")
MODEL = os.path.join(DATA, "yamnet.tflite")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"

# --- Sampling -----------------------------------------------------------------------------------
# 16 kHz is YAMNet's input rate. Nyquist = 8 kHz covers the piezo smoke-alarm band (2.8-3.6 kHz)
# and the 520 Hz low-frequency alarm with its odd harmonics up to the 13th (6.76 kHz).
SR = 16000
WIN = 15600  # classifier window: 0.975 s
HOP = 8000  # a new window every 0.5 s

# --- Test streams -------------------------------------------------------------------------------
SEED = 2026
STREAM_S = 300  # one DEMAND recording (5 min)
CHANNELS = [1, 3, 5, 7, 9, 11, 13, 15]  # DEMAND microphones used as separate streams
# Environments used to choose thresholds vs. environments never seen until the test (one per
# DEMAND category: domestic, office, public, street, transport, nature). SCAFE has no 16 kHz release.
TUNE_ENVS = ["DKITCHEN", "DWASHING", "NFIELD", "NRIVER", "OHALLWAY", "OMEETING", "PRESTO", "PSTATION",
             "SPSQUARE", "TCAR", "TMETRO"]
TEST_ENVS = ["DLIVING", "OOFFICE", "PCAFETER", "STRAFFIC", "TBUS", "NPARK"]
TARGETS_PER_STREAM = 6  # 2 per category, one every ~50 s slot
DISTRACTORS_PER_STREAM = 6  # everyday non-target sounds (half of them "hard": bells, phones, doors...)
SNRS_DB = [-5, 0, 5, 10, 20]  # target event SNR over its active frames
DISTRACTOR_SNR_DB = (0, 20)
BG_DBFS = (-50, -35)  # background level varies per stream (room loudness, mic gain)
MAX_EVENT_S = 8.0
ALARM_SEEDS = {"tune": range(0, 60), "test": range(100, 140)}

# --- Classifier decision rule (the WristEcho app's SoundDetector) ------------------------------
CATEGORIES = {
    "FIRE_ALARM": dict(labels=["Smoke detector, smoke alarm", "Fire alarm", "Alarm"], threshold=0.30, needed=2,
                       veto=["Ringtone", "Telephone", "Telephone bell ringing"]),
    "SIREN": dict(labels=["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
                          "Fire engine, fire truck (siren)", "Emergency vehicle"], threshold=0.30, needed=2, veto=[]),
    "DOOR": dict(labels=["Knock", "Doorbell", "Ding-dong"], threshold=0.30, needed=1, veto=[]),
}
ORDER = list(CATEGORIES)  # priority: most urgent first
COOLDOWN_S = 20.0
HIT_MARGIN_S = 1.0  # a detection counts for an event from its onset to 1 s after it ends

# --- DSP front end ------------------------------------------------------------------------------
FRAME = 320  # 20 ms analysis frame
NFFT = 512
HPF_HZ, HPF_ORDER = 100.0, 2  # Butterworth high-pass on the gate path: removes DC, rumble, handling
N_BANDS, BAND_LO, BAND_HI = 16, 100.0, 7600.0  # log-spaced bands for the noise-floor tracker
BAND_SMOOTH_S = 0.06  # 1st-order IIR smoothing of band power before tracking (tames FFT-bin fluctuations)
FLOOR_RISE_DB_S = 3.0
CFAR_REF_S = 10.0  # OS-CFAR reference: median novelty of the previous 10 s
# Gate variants compared on the tuning environments; the best one at the 1 % budget is used.
GATE_VARIANTS = {
    "v1": dict(smooth_s=0.06, rank=1, cfar=False),  # the first version
    "smooth2": dict(smooth_s=0.15, rank=2, cfar=False),  # longer smoothing + two bands must rise
    "v1+cfar": dict(smooth_s=0.06, rank=1, cfar=True),
    "smooth2+cfar": dict(smooth_s=0.15, rank=2, cfar=True),
}  # floor follows a quieter band at once and rises at most 3 dB/s (minimum tracking)
# Goertzel probes, fixed from the alarm literature (not tuned on data): piezo smoke alarms ring at
# about 3.1 kHz (2.8-3.6 kHz), low-frequency alarms use a 520 Hz square wave (520, 1560, 2600 Hz).
PIEZO_PROBES = np.arange(2800, 3601, 50)
SQUARE_PROBES = (520, 1560, 2600)
PROBE_STEP = 50  # Hz: one DFT bin of a 320-sample frame at 16 kHz
ALARM_V3 = dict(lo=2800, hi=3600, ratio=0.25, rule="t3_strict")  # first design (v3)
# Detector redesign (eval/alarm_design.py), searched on the design half of the real recordings:
ALARM_GRID = dict(bands=[(2800, 3600), (2500, 4000), (2200, 4500), (2000, 5000)], ratios=[0.15, 0.25, 0.35],
                  rules=["t3_strict", "t3", "periodic4", "periodic6", "t3_strict|periodic4", "t3_strict|periodic6"])

# --- Sweeps and calibration ---------------------------------------------------------------------
DUTY_K = [1, 2, 3, 4, 6, 8]  # F1: classify every k-th window
FIXED_DBFS = np.arange(-66, -19, 2.0)  # F2: absolute level threshold
NOVELTY_DB = np.arange(0, 30.1, 1.0)  # F3: dB above the floor (or above the CFAR median)
HANGOVER_S = [0, 1, 2]  # keep the classifier awake this long after a trigger
BUDGETS = [0.01, 0.03, 0.05]  # max share of F0's detections lost *because the gate slept* (tuning)
HOLD_S = 2.0  # an N-window run survives skipped windows if classified windows are <= 2 s apart
BOOTSTRAP = 2000
THRESHOLD_GRID = np.round(np.arange(0.05, 0.501, 0.025), 3)  # tuned-baseline search (must contain 0.30)

WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))
