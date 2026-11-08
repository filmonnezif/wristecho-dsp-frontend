"""The trained classifier head: a small network on YAMNet's 1024-d embedding (YAMNet stays frozen).

Stored as plain arrays (results/head.npz; demo/head.mat for MATLAB): input standardisation, then
dense layers with ReLU between them and a softmax over config.HEAD_CLASSES. A logistic-regression
head is the one-layer case. eval/train_head.py trains it; this module only runs it.
"""
import os

import numpy as np

import config as C

PATH = os.path.join(C.RESULTS, "head.npz")
_head = None


def load(path=PATH):
    global _head
    if _head is None or path != PATH:
        z = np.load(path)
        n = int(z["layers"])
        h = dict(mu=z["mu"], sd=z["sd"], W=[z[f"W{i}"] for i in range(n)], b=[z[f"b{i}"] for i in range(n)],
                 thresholds=dict(zip(C.ORDER, z["thresholds"].tolist())) if "thresholds" in z else None)
        if path != PATH:
            return h
        _head = h
    return _head


def available():
    return os.path.exists(PATH)


def probs(emb, h=None):
    """(n, len(HEAD_CLASSES)) class probabilities for embeddings (n, 1024)."""
    h = h or load()
    a = (np.asarray(emb, np.float32) - h["mu"]) / h["sd"]
    for i, (w, b) in enumerate(zip(h["W"], h["b"])):
        a = a @ w + b
        if i < len(h["W"]) - 1:
            a = np.maximum(a, 0)
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def category_scores(emb, h=None):
    """(n_windows, categories, 2) like yamnet.category_scores: the head's probability per
    category; the veto column is 0 (the head has learnt the look-alikes as "other")."""
    p = probs(emb, h)
    out = np.zeros((len(p), len(C.ORDER), 2), np.float32)
    out[:, :, 0] = p[:, [C.HEAD_CLASSES.index(c) for c in C.ORDER]]
    return out


def macs(h=None):
    h = h or load()
    return int(sum(w.size for w in h["W"]) + h["mu"].size)
