"""Step 3: every figure of the paper, from results/results.json (+ one re-rendered test stream)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from data import streams  # noqa: E402
from dsp import frontend  # noqa: E402
from eval import evaluate as E  # noqa: E402

plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"F0": "#555555", "F0+A": "#999999", "F1": "#E69F00", "F2": "#56B4E9", "F3": "#009E73", "F4": "#D55E00"}
NAME = {"F0": "F0 always on", "F0+A": "F0 + alarm DSP", "F1": "F1 every k-th", "F2": "F2 fixed gate",
        "F3": "F3 adaptive gate", "F4": "F4 proposed"}


def save(fig, name):
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, name + ".pdf"))
    fig.savefig(os.path.join(C.FIGURES, name + ".png"))
    plt.close(fig)


def pareto(r):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, pts in r["pareto_test"].items():
        pts = sorted(pts, key=lambda p: p["duty"])
        ax.plot([100 * p["duty"] for p in pts], [100 * p["recall"] for p in pts], "-o", ms=2, lw=1, color=COL[k], label=NAME[k])
    for k, f in r["frontends"].items():
        ax.plot(100 * f["duty_events"], 100 * f["recall"], "*" if k == "F4" else "s", ms=9 if k == "F4" else 5,
                color=COL[k], mec="k", mew=0.5, zorder=5, label=None if k in ("F1", "F2", "F3") else NAME[k])
    ax.set_xlabel("classifier duty on event streams (%)")
    ax.set_ylabel("recall, all targets (%)")
    ax.legend(fontsize=6, loc="lower right")
    save(fig, "fig_pareto")


def idle_duty(r):
    keys = [k for k in ("F1", "F2", "F3") if k in r["frontends"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        d = r["frontends"][k]["duty_idle_by_env"]
        ax.bar(np.arange(len(C.TEST_ENVS)) + i * w, [100 * d.get(e, np.nan) for e in C.TEST_ENVS], w, color=COL[k], label=NAME[k])
    ax.set_xticks(np.arange(len(C.TEST_ENVS)) + w * (len(keys) - 1) / 2, C.TEST_ENVS, rotation=30)
    ax.set_ylabel("classifier duty, background only (%)")
    ax.legend(fontsize=6)
    save(fig, "fig_idle_duty")


def recall_snr(r):
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True)
    for ax, cat in zip(axs, C.ORDER):
        for k in ("F0", "F3", "F4"):
            if k not in r["frontends"]:
                continue
            f = r["frontends"][k]
            ys = [f["by_cat_snr"][cat][str(s)] for s in C.SNRS_DB]
            ax.plot(C.SNRS_DB, [100 * y if y is not None else np.nan for y in ys], "-o", ms=3, color=COL[k], label=NAME[k])
        ax.set_title(cat.replace("_", " ").lower())
        ax.set_xlabel("SNR (dB)")
    axs[0].set_ylabel("recall (%)")
    axs[0].legend(fontsize=6)
    save(fig, "fig_recall_snr")


def filters():
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.2))
    sos = frontend.hpf_sos()
    w, h = signal.sosfreqz(sos, worN=4096, fs=C.SR)
    axs[0].semilogx(w[1:], 20 * np.log10(np.abs(h[1:]) + 1e-12), color=COL["F3"])
    axs[0].axvline(C.HPF_HZ, ls="--", c="k", lw=0.7)
    axs[0].set(xlim=(10, 8000), ylim=(-60, 3), xlabel="Hz", ylabel="dB", title="HPF: Butterworth, 2nd order, 100 Hz")
    z, p, _ = signal.sos2zpk(sos)
    t = np.linspace(0, 2 * np.pi, 400)
    axs[1].plot(np.cos(t), np.sin(t), "k", lw=0.5)
    axs[1].plot(z.real, z.imag, "o", mfc="none", c=COL["F3"], label="zeros (HPF)")
    axs[1].plot(p.real, p.imag, "x", c=COL["F3"], label="poles (HPF)")
    wg = 2 * np.pi * 3100 / C.SR
    axs[1].plot(np.cos([wg, -wg]), np.sin([wg, -wg]), "x", c=COL["F4"], label="poles (Goertzel 3.1 kHz)")
    axs[1].set(aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), title="pole-zero plot")
    axs[1].legend(fontsize=5, loc="lower left")
    f = np.linspace(2000, 4200, 2000)
    n = np.arange(C.FRAME)
    win = np.hanning(C.FRAME)
    for f0 in (3000, 3100, 3200):
        resp = [abs(np.sum(win * np.exp(1j * 2 * np.pi * (ff - f0) * n / C.SR))) for ff in f]
        axs[2].plot(f, 20 * np.log10(np.array(resp) / max(resp)), lw=0.8)
    axs[2].set(ylim=(-60, 3), xlabel="Hz", ylabel="dB", title="Goertzel probes (N = 320, Hann)")
    save(fig, "fig_filters")


def example(r):
    fe = r["frontends"].get("F4", {}).get("frontend")
    spec = next(s for s in streams.specs() if s["split"] == "test" and s["env"] == "DLIVING" and s["kind"] == "events")
    x, events = streams.render(spec)
    level, novelty = frontend.gate_features(x)
    t_fr = np.arange(len(novelty)) * C.FRAME / C.SR
    fig, axs = plt.subplots(3, 1, figsize=(7, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 0.5]))
    f, t, S = signal.spectrogram(x, C.SR, nperseg=512, noverlap=256)
    axs[0].pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-40, rasterized=True)
    axs[0].set_ylabel("kHz")
    for e in events:
        if e["role"] == "target":
            axs[0].axvspan(e["onset"], e["end"], color="w", alpha=0.15)
            axs[0].text(e["onset"], 7.3, f"{e['category'].lower()} {e['snr']:.0f} dB", color="w", fontsize=5)
    axs[1].plot(t_fr, novelty, lw=0.4, color=COL["F3"])
    if fe:
        axs[1].axhline(fe[1], color="k", ls="--", lw=0.7)
        nov_w = frontend.per_window(novelty, (len(x) - C.WIN) // C.HOP + 1)
        awake = E.hang(nov_w > fe[1], fe[2])
        tw = (C.WIN + np.arange(len(awake)) * C.HOP) / C.SR
        axs[2].fill_between(tw, 0, awake, step="pre", color=COL["F4"], alpha=0.8)
        axs[2].set_title(f"classifier awake {100 * awake.mean():.0f} % of the time", fontsize=7)
    axs[1].set_ylabel("novelty (dB)")
    axs[2].set(yticks=[], xlabel="time (s)")
    save(fig, "fig_example")


def main():
    r = json.load(open(os.path.join(C.RESULTS, "results.json")))
    filters()
    pareto(r)
    idle_duty(r)
    recall_snr(r)
    example(r)
    print("  figures ->", C.FIGURES)


if __name__ == "__main__":
    main()
