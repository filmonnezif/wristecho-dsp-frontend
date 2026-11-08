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
# 10 targets per stream, one every 30 s slot: 2 sirens, 2 car horns, 2 baby cries, 2 door sounds,
# 1 synthetic T-3 fire alarm and 1 real fire/smoke alarm recording.
STREAM_EVENTS = ["SIREN", "SIREN", "CAR_HORN", "CAR_HORN", "BABY_CRY", "BABY_CRY", "DOOR", "DOOR",
                 "FIRE_ALARM:t3", "FIRE_ALARM:real"]
TARGETS_PER_STREAM = len(STREAM_EVENTS)
DISTRACTORS_PER_STREAM = 8  # everyday non-target sounds (half of them "hard": bells, phones, beeps...)
SNRS_DB = [-5, 0, 5, 10, 20]  # target event SNR over its active frames
DISTRACTOR_SNR_DB = (0, 20)
BG_DBFS = (-50, -35)  # background level varies per stream (room loudness, mic gain)
MAX_EVENT_S = 8.0
ALARM_SEEDS = {"tune": range(0, 60), "test": range(100, 140), "train": range(200, 400)}
# Classifier-head training streams: the tuning environments' even microphones (the odd ones are the
# tuning streams), two renderings each, from the tuning clips set aside for training (TRAIN_SHARE of
# them, by clip id). Thresholds are then chosen on the tuning streams, whose clips the head never saw.
TRAIN_CHANNELS = [2, 4, 6, 8, 10, 12, 14, 16]
TRAIN_COPIES = 2
TRAIN_SHARE = 0.7

# --- Classifier decision rule (the WristEcho app's SoundDetector) ------------------------------
CATEGORIES = {
    "FIRE_ALARM": dict(labels=["Smoke detector, smoke alarm", "Fire alarm", "Alarm"], threshold=0.30, needed=2,
                       veto=["Ringtone", "Telephone", "Telephone bell ringing"]),
    "SIREN": dict(labels=["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
                          "Fire engine, fire truck (siren)", "Emergency vehicle"], threshold=0.30, needed=2, veto=[]),
    "CAR_HORN": dict(labels=["Vehicle horn, car horn, honking", "Air horn, truck horn"], threshold=0.30, needed=1, veto=[]),
    "BABY_CRY": dict(labels=["Baby cry, infant cry"], threshold=0.30, needed=2, veto=[]),
    "DOOR": dict(labels=["Knock", "Doorbell", "Ding-dong"], threshold=0.30, needed=1, veto=[]),
}
ORDER = list(CATEGORIES)  # priority: most urgent first

# --- Trained classifier head ---------------------------------------------------------------------
# A small classifier on YAMNet's 1024-d embedding (YAMNet itself stays frozen), trained on the
# training streams. A window is labelled with a target category when that target's sound (frames
# within LABEL_DB of its own peak) covers at least LABEL_MIN_S of the window; otherwise "other".
HEAD_CLASSES = ORDER + ["OTHER"]
LABEL_DB, LABEL_MIN_S = 30.0, 0.06
# "+yamnet": the same model also reading YAMNet's own category scores (stacked head)
HEAD_MODELS = {"logreg": dict(C=[0.001, 0.01, 0.1]), "mlp": dict(hidden=[128, 256], alpha=[1e-4, 1e-3]),
               "logreg+yamnet": dict(C=[0.001, 0.01, 0.1]), "mlp+yamnet": dict(hidden=[128, 256], alpha=[1e-4, 1e-3])}
FA_BUDGET_PER_H = 0.3  # per category: thresholds of the head (and tuned YAMNet) keep tuning FA/h below this
HEAD_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
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
    # two floors: the fast one (3 dB/s) plus a slow one (0.3 dB/s) for sounds that swell slowly
    "two-floor+6": dict(smooth_s=0.06, rank=1, cfar=False, slow_rise=0.3, slow_offset=6),
    "two-floor+10": dict(smooth_s=0.06, rank=1, cfar=False, slow_rise=0.3, slow_offset=10),
    "two-floor+14": dict(smooth_s=0.06, rank=1, cfar=False, slow_rise=0.3, slow_offset=14),
}
VARIANT_BUDGET = 0.05  # the gate statistic is chosen at the deployed (proposed) budget
# Knock detector (DSP, real knocks): impulse threshold (dB), impulses needed, and the YAMNet
# door score required within +-2 s (0 = DSP alone). Chosen on tuning streams: most door recall
# with no more false alarms than the system without it.
KNOCK_GRID = dict(kappa=[6, 8, 10, 12, 15, 18], hits=[2, 3, 4], agree=[0.0, 0.05, 0.1, 0.2])
ALARM_AGREE = 0.1  # DSP fire alarm accepted only if YAMNet's fire score reaches this within +-2 s
AGREE_S = 2.0  # floor follows a quieter band at once and rises at most 3 dB/s (minimum tracking)
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
THRESHOLD_GRID = HEAD_THRESHOLDS  # tuned-baseline search (must contain 0.30)

WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))
