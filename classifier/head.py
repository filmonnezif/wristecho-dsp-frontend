"""The trained classifier head: a small network on YAMNet's 1024-d embedding (YAMNet stays frozen).

Stored as plain arrays (results/head.npz; demo/head.mat for MATLAB): input standardisation, then
dense layers with ReLU between them and a softmax over config.HEAD_CLASSES. A logistic-regression
head is the one-layer case. A "stacked" head also reads YAMNet's own category scores (as logits),
so it refines YAMNet's answer instead of replacing it. eval/train_head.py trains it; this module
only runs it.
"""
import os

import numpy as np

import config as C

PATH = os.path.join(C.RESULTS, "head.npz")  # the YAMNet head; other backbones: head_<backbone>.npz
_head = None


def path_for(backbone):
    return PATH if backbone == "yamnet" else os.path.join(C.RESULTS, f"head_{backbone}.npz")


def labels_for(backbone):
    """Class names of a backbone's scores (EfficientAT's are saved next to its cache by eval/extract_eat.py)."""
    if backbone == "yamnet":
        from classifier import yamnet
        return yamnet.labels()
    import json
    return json.load(open(os.path.join(C.CACHE, f"eat_{backbone}", "labels.json")))


def load(path=PATH):
    global _head
    if _head is None or path != PATH:
        z = np.load(path)
        n = int(z["layers"])
        h = dict(mu=z["mu"], sd=z["sd"], W=[z[f"W{i}"] for i in range(n)], b=[z[f"b{i}"] for i in range(n)],
                 thresholds=dict(zip(C.ORDER, z["thresholds"].tolist())) if "thresholds" in z else None,
                 stack=bool(z["stack"]) if "stack" in z else False,
                 backbone=str(z["backbone"]) if "backbone" in z else "yamnet")
        if path != PATH:
            return h
        _head = h
    return _head


def available():
    return os.path.exists(PATH)


def features(emb, stock=None, stack=False):
    """The head's input: the embedding, plus (stacked) the logits of YAMNet's category scores
    (n, categories), as yamnet.category_scores(...)[:, :, 0]."""
    emb = np.asarray(emb, np.float32)
    if not stack:
        return emb
    p = np.clip(np.asarray(stock, np.float32), 1e-4, 1 - 1e-4)
    return np.c_[emb, np.log(p / (1 - p))]


def probs(x, h=None):
    """(n, len(HEAD_CLASSES)) class probabilities for head inputs x (see features)."""
    h = h or load()
    a = (np.asarray(x, np.float32) - h["mu"]) / h["sd"]
    for i, (w, b) in enumerate(zip(h["W"], h["b"])):
        a = a @ w + b
        if i < len(h["W"]) - 1:
            a = np.maximum(a, 0)
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def category_scores(emb, scores, h=None):
    """(n_windows, categories, 2) like yamnet.category_scores, from the embeddings and YAMNet's 521
    scores: the head's probability per category; the veto column is 0 (the head has learnt the
    look-alikes as "other")."""
    from classifier import yamnet
    h = h or load()
    if h.get("stack") is None:
        h["stack"] = False
    p = probs(features(emb, yamnet.category_scores(scores, labels_for(h["backbone"]))[:, :, 0], h["stack"]), h)
    out = np.zeros((len(p), len(C.ORDER), 2), np.float32)
    out[:, :, 0] = p[:, [C.HEAD_CLASSES.index(c) for c in C.ORDER]]
    return out


def macs(h=None):
    h = h or load()
    return int(sum(w.size for w in h["W"]) + h["mu"].size)
