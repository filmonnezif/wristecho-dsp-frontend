"""YAMNet (MediaPipe yamnet.tflite) on 0.975 s windows every 0.5 s, and the alert decision rule."""
import zipfile

import numpy as np

import config as C

_interp = None
_labels = None
_emb_index = None


def labels():
    global _labels
    if _labels is None:
        _labels = zipfile.ZipFile(C.MODEL).read("yamnet_label_list.txt").decode().splitlines()
    return _labels


def _model():
    """The interpreter keeps every intermediate tensor, so the embedding (the 1024-d input of the
    final fully connected layer, after global average pooling) can be read with the scores."""
    global _interp, _emb_index
    if _interp is None:
        from ai_edge_litert.interpreter import Interpreter
        _interp = Interpreter(C.MODEL, num_threads=1, experimental_preserve_all_tensors=True)
        _interp.allocate_tensors()
        shape = {t["index"]: list(t["shape"]) for t in _interp.get_tensor_details()}
        fc = [op for op in _interp._get_ops_details() if op["op_name"] == "FULLY_CONNECTED"
              and shape[op["outputs"][0]][-1] == len(labels())]
        _emb_index = int(fc[-1]["inputs"][0])
        assert shape[_emb_index][-1] == 1024, shape[_emb_index]
    return _interp


def macs_per_window():
    """Multiply-accumulates of one YAMNet inference, counted from the model graph
    (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED; the mel front end inside the model is ignored)."""
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(C.MODEL, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    shape = {t["index"]: t["shape"] for t in it.get_tensor_details()}
    total = 0
    for op in it._get_ops_details():
        w, o = shape[op["inputs"][1]] if len(op["inputs"]) > 1 else None, shape[op["outputs"][0]]
        if op["op_name"] == "CONV_2D":
            total += int(np.prod(o) * np.prod(w[1:]))
        elif op["op_name"] == "DEPTHWISE_CONV_2D":
            total += int(np.prod(o) * w[1] * w[2])
        elif op["op_name"] == "FULLY_CONNECTED":
            total += int(np.prod(o) * w[1])
    return total


def n_windows(n_samples):
    return (n_samples - C.WIN) // C.HOP + 1


def window_outputs(x):
    """(n_windows, 521) scores and (n_windows, 1024) embeddings; window k covers samples
    [k*HOP, k*HOP + WIN)."""
    m = _model()
    out = m.get_output_details()[0]["index"]
    n = n_windows(len(x))
    scores = np.empty((n, len(labels())), np.float32)
    emb = np.empty((n, 1024), np.float32)
    for k in range(n):
        m.set_tensor(m.get_input_details()[0]["index"], x[k * C.HOP:k * C.HOP + C.WIN].astype(np.float32))
        m.invoke()
        scores[k] = m.get_tensor(out)[0]
        emb[k] = m.get_tensor(_emb_index).reshape(-1)
    return scores, emb


def window_scores(x):
    """(n_windows, 521) scores."""
    return window_outputs(x)[0]


def category_scores(scores, label_list=None):
    """(n_windows, categories, 2): best label score and best veto-label score per category.
    label_list: the scores' class names (default YAMNet's; EfficientAT uses AudioSet's 527)."""
    idx = {l: i for i, l in enumerate(label_list or labels())}
    out = np.zeros((len(scores), len(C.ORDER), 2), np.float32)
    for j, c in enumerate(C.ORDER):
        d = C.CATEGORIES[c]
        out[:, j, 0] = scores[:, [idx[l] for l in d["labels"] if l in idx]].max(axis=1)
        if any(l in idx for l in d["veto"]):
            out[:, j, 1] = scores[:, [idx[l] for l in d["veto"] if l in idx]].max(axis=1)
    return out


def detect(cat, awake=None, extra=(), thresholds=None, siren_veto=False):
    """The app's rule: a category fires when its score >= threshold (and beats its veto labels)
    in `needed` consecutive classified windows, with a 20 s cooldown; most urgent first.
    awake: bool per window (None = always). A sleeping window is not classified; a run of
    `needed` windows counts classified windows and survives skipped ones while consecutive
    classified windows are at most HOLD_S apart (so every-k-th-window duty cycling can still meet
    a 2-window rule; with no skipping this is exactly the app's rule).
    extra: [(time, category)] detections from the DSP alarm detector, merged under the same cooldown.
    thresholds: {category: score threshold} overriding the app's 0.30 (tuned-baseline ablation).
    siren_veto: a fire alarm needs its score to beat the siren score in the same window (YAMNet
    also scores "Alarm" on sirens, and the fire alarm has priority).
    Returns [(time s, category)]."""
    times = (C.WIN + np.arange(len(cat)) * C.HOP) / C.SR
    runs = [0] * len(C.ORDER)
    last = {c: -1e9 for c in C.ORDER}
    fired = []
    ex = sorted(extra)
    e = 0
    last_classified = -1e9
    thr = np.array([(thresholds or {}).get(c, C.CATEGORIES[c]["threshold"]) for c in C.ORDER])
    ok = (cat[:, :, 0] >= thr) & (cat[:, :, 1] < cat[:, :, 0])
    if siren_veto:
        f, sr = C.ORDER.index("FIRE_ALARM"), C.ORDER.index("SIREN")
        ok[:, f] &= cat[:, f, 0] > cat[:, sr, 0]
    ok_all = ok.tolist()  # per window, per category
    needed = [C.CATEGORIES[c]["needed"] for c in C.ORDER]
    times = times.tolist()
    awake = awake.tolist() if awake is not None else None
    for k, t in enumerate(times):
        while e < len(ex) and ex[e][0] <= t:
            te, ce = ex[e]
            if te - last[ce] >= C.COOLDOWN_S:
                last[ce] = te
                fired.append((float(te), ce))
            e += 1
        if awake is not None and not awake[k]:
            continue
        if t - last_classified > C.HOLD_S + 1e-6:
            runs = [0] * len(C.ORDER)
        last_classified = t
        winner = None
        for j, ok in enumerate(ok_all[k]):
            runs[j] = runs[j] + 1 if ok else 0
            if winner is None and runs[j] >= needed[j] and t - last[C.ORDER[j]] >= C.COOLDOWN_S:
                winner = j
        if winner is not None:
            c = C.ORDER[winner]
            last[c] = t
            runs[winner] = 0
            fired.append((float(t), c))
    for te, ce in ex[e:]:
        if te - last[ce] >= C.COOLDOWN_S:
            last[ce] = te
            fired.append((float(te), ce))
    return fired
