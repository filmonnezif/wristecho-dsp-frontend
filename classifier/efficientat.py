"""EfficientAT (Schmid, Koutini and Widmer, ICASSP 2023): MobileNetV3 audio taggers trained on AudioSet by
knowledge distillation from transformers. Used frozen as an alternative backbone to YAMNet: the same
0.975 s windows every 0.5 s, resampled to the models' 32 kHz; 527 AudioSet scores and the pooled
embedding per window. Runs on a GPU when one is present (Colab).
Code and weights: https://github.com/fschmid56/EfficientAT (cloned into DATA_DIR on first use).
"""
import os
import subprocess
import sys

import numpy as np
from scipy import signal

import config as C

REPO = os.path.join(C.DATA, "EfficientAT")
MODELS = ["mn04_as", "mn10_as"]
_cache = {}


def _load(name):
    if name not in _cache:
        import torch
        if not os.path.isdir(REPO):
            subprocess.run(["git", "clone", "-q", "https://github.com/fschmid56/EfficientAT", REPO], check=True)
        here = os.getcwd()
        os.chdir(REPO)  # the repository loads its label list and stores weights relative to itself
        sys.path.insert(0, REPO)
        try:
            from helpers.utils import NAME_TO_WIDTH, labels
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = get_model(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).to(dev).eval()
            mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320).to(dev).eval()
        finally:
            os.chdir(here)
        _cache[name] = (model, mel, dev, list(labels))
    return _cache[name]


def labels(name="mn04_as"):
    """The 527 AudioSet class names, in the models' output order (no model or GPU needed)."""
    if name in _cache:
        return _cache[name][3]
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "-q", "https://github.com/fschmid56/EfficientAT", REPO], check=True)
    here = os.getcwd()
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    try:
        from helpers.utils import labels as names
    finally:
        os.chdir(here)
    return list(names)


def window_outputs(x, name, batch=256):
    """(n_windows, 527) sigmoid scores and (n_windows, D) embeddings for 16 kHz audio x."""
    import torch
    model, mel, dev, _ = _load(name)
    y = signal.resample_poly(x, 2, 1).astype(np.float32)
    n = (len(x) - C.WIN) // C.HOP + 1
    wins = np.stack([y[2 * k * C.HOP:2 * k * C.HOP + 2 * C.WIN] for k in range(n)])
    scores, emb = [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            spec = mel(torch.from_numpy(wins[i:i + batch]).to(dev))
            logits, feat = model(spec.unsqueeze(1))
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            emb.append(feat.float().reshape(len(spec), -1).cpu().numpy())
    return np.concatenate(scores), np.concatenate(emb)


def macs_per_window(name):
    """Multiply-accumulates of one 0.975 s window through the network (mel front end not counted,
    as for YAMNet), from PyTorch's FLOP counter (FLOPs / 2)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    model, mel, dev, _ = _load(name)
    with torch.no_grad():
        spec = mel(torch.zeros(1, 2 * C.WIN, device=dev))
        fc = FlopCounterMode(display=False)
        with fc:
            model(spec.unsqueeze(1))
    return int(fc.get_total_flops() // 2)
