"""Step 1c: train the classifier head on YAMNet's embeddings (YAMNet itself stays frozen).

Data: the training streams (cache/v6train: tuning environments' even microphones, "train" clips).
Each 0.975 s window is labelled with the target sounding in it (config.LABEL_DB / LABEL_MIN_S) or
"other" (background, distractors, look-alike beeps). Models (config.HEAD_MODELS): multinomial
logistic regression (balanced class weights) and a one-hidden-layer MLP (balanced resampling).
Model selection: mean average precision over the target classes on the tuning streams' windows
("calib" clips, never trained on). Thresholds: per category, the lowest whose false alarms on the
tuning streams stay within FA_BUDGET_PER_H (the same rule as the tuned YAMNet baseline).
The test streams are only scored at the end (window-level average precision, for the paper).
Writes results/head.npz (used by the pipeline), results/head.mat (MATLAB demo), results/head.json.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import head, yamnet  # noqa: E402
from eval.extract import DIRS  # noqa: E402

OTHER = C.HEAD_CLASSES.index("OTHER")


def window_labels(events, n):
    """Class index per window: the most urgent target sounding for at least LABEL_MIN_S of it."""
    t0 = np.arange(n) * C.HOP / C.SR
    t1 = t0 + C.WIN / C.SR
    y = np.full(n, OTHER)
    for e in sorted((e for e in events if e["role"] == "target"), key=lambda e: -C.ORDER.index(e["category"])):
        cover = sum(np.clip(np.minimum(t1, b) - np.maximum(t0, a), 0, None) for a, b in e["active"])
        y[np.asarray(cover) >= C.LABEL_MIN_S] = C.HEAD_CLASSES.index(e["category"])
    return y


def _read(arg):
    p, backbone = arg
    z = np.load(p)
    meta = json.loads(str(z["meta"]))
    n = int(z["n"])
    if backbone != "yamnet":  # the same stream's EfficientAT outputs
        z = np.load(os.path.join(C.CACHE, f"eat_{backbone}", os.path.basename(p)))
    return z["emb"].astype(np.float32), window_labels(meta["events"], n), \
        yamnet.category_scores(z["scores"].astype(np.float32), head.labels_for(backbone))[:, :, 0], meta["split"]


def read(paths, backbone="yamnet"):
    with Pool(C.WORKERS) as pool:
        r = pool.map(_read, [(p, backbone) for p in paths])
    return np.concatenate([a for a, *_ in r]), np.concatenate([b for _, b, *_ in r]), \
        np.concatenate([c for *_, c, _ in r]), [d for *_, d in r]


def mean_ap(y, p):
    from sklearn.metrics import average_precision_score
    ap = {c: float(average_precision_score(y == i, p[:, i])) for i, c in enumerate(C.ORDER) if (y == i).any()}
    return float(np.mean(list(ap.values()))), ap


def fit(kind, params, X, y, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if kind.startswith("logreg"):
        m = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=500)
        m.fit(X, y)
        return [m.coef_.T.astype(np.float32)], [m.intercept_.astype(np.float32)]
    counts = np.bincount(y, minlength=len(C.HEAD_CLASSES))
    top = counts[:OTHER].max()
    idx = np.concatenate([rng.choice(np.flatnonzero(y == i), top if i != OTHER else min(counts[i], 3 * top),
                                     replace=i != OTHER) for i in range(len(C.HEAD_CLASSES))])
    m = MLPClassifier(hidden_layer_sizes=(params["hidden"],), alpha=params["alpha"], early_stopping=True,
                      max_iter=60, random_state=C.SEED)
    m.fit(X[idx], y[idx])
    return [w.astype(np.float32) for w in m.coefs_], [b.astype(np.float32) for b in m.intercepts_]


def save(path, mu, sd, W, b, stack, thresholds=None, backbone="yamnet"):
    arrs = dict(mu=mu, sd=sd, layers=len(W), stack=stack, backbone=backbone, **{f"W{i}": w for i, w in enumerate(W)},
                **{f"b{i}": v for i, v in enumerate(b)})
    if thresholds:
        arrs["thresholds"] = np.array([thresholds[c] for c in C.ORDER], np.float32)
    np.savez(path, **arrs)


def main():
    from classifier import efficientat
    for backbone in ["yamnet"] + [m for m in efficientat.MODELS if os.path.isdir(os.path.join(C.CACHE, f"eat_{m}"))]:
        print(f"  --- backbone {backbone}", flush=True)
        train(backbone)


def train(backbone):
    t = time.time()
    rng = np.random.default_rng(C.SEED)
    E, y, stock, _ = read(sorted(glob.glob(os.path.join(DIRS[True], "*.npz"))), backbone)
    Ev, yv, stock_v, _ = read(sorted(glob.glob(os.path.join(DIRS[False], "tune-*.npz"))), backbone)
    print(f"  training windows {len(y):,} {np.bincount(y, minlength=len(C.HEAD_CLASSES)).tolist()}, "
          f"tuning windows {len(yv):,} ({time.time() - t:.0f} s)", flush=True)
    table, best = [], None
    for kind, grid in C.HEAD_MODELS.items():
        stack = kind.endswith("+scores")
        X, Xv = head.features(E, stock, stack), head.features(Ev, stock_v, stack)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs, Xvs = (X - mu) / sd, (Xv - mu) / sd
        keys = list(grid)
        for vals in np.array(np.meshgrid(*grid.values()), dtype=object).T.reshape(-1, len(keys)):
            params = {k: (int(v) if k == "hidden" else float(v)) for k, v in zip(keys, vals)}
            W, b = fit(kind, params, Xs, y, rng)
            h = dict(mu=np.zeros_like(mu), sd=np.ones_like(sd), W=W, b=b)
            m, ap = mean_ap(yv, head.probs(Xvs, h))
            table.append(dict(kind=kind, params=params, tune_map=m, tune_ap=ap))
            print(f"  {kind} {params}: tuning mAP {m:.3f} {({k: round(v, 3) for k, v in ap.items()})} "
                  f"({time.time() - t:.0f} s)", flush=True)
            if best is None or m > best[0]:
                best = (m, kind, params, W, b, mu, sd, stack)
    _, kind, params, W, b, mu, sd, stack = best
    os.makedirs(C.RESULTS, exist_ok=True)
    path = head.path_for(backbone)
    save(path, mu, sd, W, b, stack, backbone=backbone)
    head._head = None
    thr = None
    if backbone == "yamnet":  # the pipeline's head (eval/evaluate.py); other backbones are tuned in eval/improve.py
        from eval import evaluate
        tune = [s for s in evaluate.load() if s["split"] == "tune"]
        thr = evaluate.tune_thresholds(tune, "head")
        save(path, mu, sd, W, b, stack, thr, backbone)
        head._head = None
    from scipy.io import savemat
    savemat(path[:-4] + ".mat", dict(mu=mu, sd=sd, stack=stack, classes=np.array(C.HEAD_CLASSES, dtype=object),
                                     categories=np.array(C.ORDER, dtype=object),
                                     thresholds=np.array([thr[c] for c in C.ORDER] if thr else [np.nan] * len(C.ORDER)),
                                                     **{f"W{i}": w for i, w in enumerate(W)},
                                                     **{f"b{i}": v.reshape(1, -1) for i, v in enumerate(b)}))
    # test windows, scored once at the end: the stock label score vs the head's probability
    Et, yt, stock_t, _ = read(sorted(glob.glob(os.path.join(DIRS[False], "test-*.npz"))), backbone)
    pt = head.probs(head.features(Et, stock_t, stack), dict(mu=mu, sd=sd, W=W, b=b))
    out = dict(chosen=dict(kind=kind, params=params, macs=int(sum(w.size for w in W) + mu.size)), selection=table, thresholds=thr,
               train_counts=dict(zip(C.HEAD_CLASSES, np.bincount(y, minlength=len(C.HEAD_CLASSES)).tolist())),
               test_counts=dict(zip(C.HEAD_CLASSES, np.bincount(yt, minlength=len(C.HEAD_CLASSES)).tolist())),
               test_ap_head=mean_ap(yt, pt), test_ap_stock=mean_ap(yt, np.c_[stock_t, np.zeros(len(yt))]),
               tune_ap_stock=mean_ap(yv, np.c_[stock_v, np.zeros(len(yv))]))
    json.dump(out, open(path[:-4] + ".json", "w"), indent=1)
    print(f"  chosen: {kind} {params}; thresholds {thr}")
    print(f"  test window mAP: stock YAMNet {out['test_ap_stock'][0]:.3f} -> head {out['test_ap_head'][0]:.3f}")
    for c in C.ORDER:
        print(f"    {c:11s} AP stock {out['test_ap_stock'][1].get(c, float('nan')):.3f}  head {out['test_ap_head'][1].get(c, float('nan')):.3f}")


if __name__ == "__main__":
    main()
