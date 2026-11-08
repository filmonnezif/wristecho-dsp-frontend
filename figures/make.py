"""Step 3: every figure of the paper, from results/*.json (results, improve, final, alarm_design)."""
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
COL = {"F0": "#555555", "F0+A": "#999999", "F1": "#E69F00", "F2": "#56B4E9", "F3 v1": "#9AD9C4", "F3": "#009E73",
       "F4": "#D55E00"}
NAME = {"F0": "F0 always on", "F0+A": "F0 + alarm DSP", "F1": "F1 every k-th", "F2": "F2 fixed gate",
        "F3 v1": "F3 one floor", "F3": "F3 two floors", "F4": "F4 proposed"}
PROPOSED = "F4 knock@5%"  # gate + alarm DSP + knock DSP + siren veto, at the 5 % budget


def base(key):
    k = key.split("@")[0]
    return k if k in COL else k.split()[0]


def save(fig, name):
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, name + ".pdf"))
    fig.savefig(os.path.join(C.FIGURES, name + ".png"))
    plt.close(fig)


def pareto(r):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, pts in r["pareto_test"].items():
        if k not in COL:  # the head's gate curve (H3) is not part of this figure
            continue
        pts = sorted(pts, key=lambda p: p["duty"])
        ax.plot([100 * p["duty"] for p in pts], [100 * p["recall"] for p in pts], marker="o", ms=2, lw=1,
                color=COL[k], label=NAME[k], ls="--" if k == "F3 v1" else "-")
    for k in ("F0", "F2@5%", "F3@5%", PROPOSED):
        f = r["frontends"].get(k)
        if f:
            ax.plot(100 * f["duty_events"], 100 * f["recall"], "*" if k.startswith("F4") else "s",
                    ms=9 if k.startswith("F4") else 5, color=COL[base(k)], mec="k", mew=0.5, zorder=5,
                    label=NAME[k] if k in ("F0", "F0+A") else ("F4 proposed (5 % budget)" if k.startswith("F4") else None))
    ax.set(xlabel="classifier duty on event streams (%)", ylabel="recall, all targets (%)", ylim=(0, None))
    ax.legend(fontsize=5.5, loc="lower right")
    save(fig, "fig_pareto")


def idle_duty(r):
    keys = [k for k in ("F2@5%", "F3@5%") if k in r["frontends"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        d = r["frontends"][k]["duty_idle_by_env"]
        ax.bar(np.arange(len(C.TEST_ENVS)) + i * w, [100 * d.get(e, np.nan) for e in C.TEST_ENVS], w,
               color=COL[base(k)], label=NAME[base(k)])
    ax.set_xticks(np.arange(len(C.TEST_ENVS)) + w * (len(keys) - 1) / 2, C.TEST_ENVS, rotation=30)
    ax.set(ylabel="classifier duty,\nbackground only (%)", ylim=(0, 100))
    ax.legend(fontsize=6)
    save(fig, "fig_idle_duty")


def recall_snr(r):
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True)
    for ax, cat in zip(axs, C.ORDER):
        for k in ("F0", "F3@5%", PROPOSED):
            if k in r["frontends"]:
                ys = [r["frontends"][k]["by_cat_snr"][cat][str(v)] for v in C.SNRS_DB]
                ax.plot(C.SNRS_DB, [100 * y if y is not None else np.nan for y in ys], marker="o", ms=3,
                        color=COL[base(k)], label=NAME[base(k)], ls="--" if k == "F3@5%" else "-")
        ax.set(title=cat.replace("_", " ").lower(), xlabel="SNR (dB)", ylim=(0, 102))
    axs[0].set_ylabel("recall (%)")
    axs[0].legend(fontsize=6)
    save(fig, "fig_recall_snr")


def stress():
    p = os.path.join(C.RESULTS, "stress.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    names = list(r)
    det = [("yamnet", "classifier (F0)", COL["F0"]), ("t3_loose", "T-3", COL["F4"]),
           ("t3_strict", "T-3 strict", "#F0A27A"), ("dsp_v4", "DSP v4", "#8C3A12")]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = 0.8 / len(det)
    for i, (k, lab, col) in enumerate(det):
        ax.bar(np.arange(len(names)) + i * w, [100 * r[n].get(k, 0) for n in names], w, color=col, label=lab)
    ax.set_xticks(np.arange(len(names)) + 0.3, [n + ("" if r[n]["expected_fire"] else "\n(should not fire)") for n in names],
                  rotation=35, ha="right", fontsize=6)
    ax.set(ylabel="fire alarm raised (%)", ylim=(0, 105))
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    save(fig, "fig_stress")


def filters():
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.2))
    sos = frontend.hpf_sos()
    w, h = signal.sosfreqz(sos, worN=4096, fs=C.SR)
    axs[0].semilogx(w[1:], 20 * np.log10(np.abs(h[1:]) + 1e-12), color=COL["F3"])
    axs[0].axvline(C.HPF_HZ, ls="--", c="k", lw=0.7)
    axs[0].set(xlim=(10, 8000), ylim=(-60, 3), xlabel="Hz", ylabel="dB", title="High-pass filter\n(Butterworth, order 2, 100 Hz)")
    z, p, _ = signal.sos2zpk(sos)
    t = np.linspace(0, 2 * np.pi, 400)
    axs[1].plot(np.cos(t), np.sin(t), "k", lw=0.5)
    axs[1].plot(z.real, z.imag, "o", mfc="none", c=COL["F3"], label="zeros (HPF)")
    axs[1].plot(p.real, p.imag, "x", c=COL["F3"], label="poles (HPF)")
    wg = 2 * np.pi * 3100 / C.SR
    axs[1].plot(np.cos([wg, -wg]), np.sin([wg, -wg]), "x", c=COL["F4"], label="poles (Goertzel 3.1 kHz)")
    axs[1].set(aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), title="Pole-zero plot")
    axs[1].legend(fontsize=5, loc="lower left")
    f = np.linspace(2000, 4200, 2000)
    n = np.arange(C.FRAME)
    win = np.hanning(C.FRAME)
    for f0 in (3000, 3100, 3200):
        resp = [abs(np.sum(win * np.exp(1j * 2 * np.pi * (ff - f0) * n / C.SR))) for ff in f]
        axs[2].plot(f, 20 * np.log10(np.array(resp) / max(resp)), lw=0.8)
    axs[2].set(ylim=(-60, 3), xlabel="Hz", ylabel="dB", title="Goertzel probes\n(N = 320, Hann window)")
    save(fig, "fig_filters")


def alarm_character():
    """Real alarms (design half): dominant pitch vs. the Goertzel probe band, and their patterns."""
    p = os.path.join(C.RESULTS, "alarm_design.json")
    if not os.path.exists(p):
        return
    ch = json.load(open(p))["characterization"]
    f0 = [c["f0"] / 1000 for c in ch if c["f0"]]
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.2), gridspec_kw=dict(height_ratios=[1.3, 1]))
    axs[0].axvspan(C.ALARM_V3["lo"] / 1000, C.ALARM_V3["hi"] / 1000, color=COL["F4"], alpha=0.15, label="Goertzel probes")
    axs[0].hist(f0, bins=np.arange(0.75, 4.01, 0.25), color=COL["F0"], edgecolor="w")
    axs[0].set(xlabel="dominant frequency of tonal frames (kHz)", ylabel="real alarms")
    axs[0].legend(fontsize=6)
    order = ["T-3", "periodic", "continuous tone", "other", "not tonal"]
    counts = [sum(c["pattern"] == o for c in ch) for o in order]
    axs[1].barh(order[::-1], counts[::-1], color=[COL["F0"]] * 4 + [COL["F4"]])  # T-3 (top) is what the detector encodes
    axs[1].set(xlabel=f"real alarms (n = {len(ch)})")
    save(fig, "fig_alarm_character")


FINAL = "best_mn10@0.1"  # the recommended system (eval/final.py)
CURVES = [("stock", "always-on", "YAMNet, always on", COL["F0"], "o", "--"),
          ("stock", "full", "YAMNet + gate + DSP", "#56B4E9", "s", "-"),
          ("best_yamnet", "full", "YAMNet + head + gate + DSP", "#009E73", "^", "-"),
          ("best_mn10", "full", "EfficientAT + head + gate + DSP", COL["F4"], "D", "-")]


def curve():
    """Recall against false alarms per hour (test event streams) for false-alarm budgets 0.1-3 per hour."""
    p = os.path.join(C.RESULTS, "improve.json")
    if not os.path.exists(p):
        return
    imp = json.load(open(p))["curve"]
    f0 = json.load(open(os.path.join(C.RESULTS, "results.json")))["frontends"]["F0"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for clf, mode, lab, col, mk, ls in CURVES:
        pts = sorted(imp[clf].items(), key=lambda kv: float(kv[0]))
        ax.plot([v[mode]["fa_per_hour_events"] for _, v in pts], [100 * v[mode]["recall"] for _, v in pts], marker=mk, ms=3,
                lw=1, ls=ls, color=col, label=lab)
    ax.plot(f0["fa_per_hour_events"], 100 * f0["recall"], "*", ms=9, color="k", label="deployed rule (YAMNet, 0.30)")
    ax.set(xscale="log", xlabel="false alarms per hour (event streams)", ylabel="recall, 480 test events (%)", ylim=(20, 75))
    ax.legend(fontsize=5.5, loc="lower right")
    save(fig, "fig_curve")


def transfer():
    """Benchmark recall vs. real-home false alerts at each false-alarm budget (the transfer gap)."""
    p = os.path.join(C.RESULTS, "final.json")
    if not os.path.exists(p):
        return
    fin = json.load(open(p))
    imp = json.load(open(os.path.join(C.RESULTS, "improve.json")))["curve"]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.2))
    for clf, lab, col, mk in (("best_yamnet", "YAMNet + head", "#009E73", "^"), ("best_mn10", "EfficientAT + head", COL["F4"], "D")):
        bs = sorted(b for b in imp[clf] if f"{clf}@{b}" in fin)
        axs[0].plot([float(b) for b in bs], [100 * imp[clf][b]["full"]["recall"] for b in bs], marker=mk, color=col, label=lab)
        axs[1].plot([float(b) for b in bs], [fin[f"{clf}@{b}"]["sins_all"]["final_per_h"] for b in bs], marker=mk, color=col, label=lab)
    any_sys = next(v for k, v in fin.items() if "@" in k)
    axs[1].axhline(any_sys["sins_all"]["app_per_h"], color="k", ls="--", lw=0.8, label="deployed rule")
    axs[0].axhline(100 * json.load(open(os.path.join(C.RESULTS, "results.json")))["frontends"]["F0"]["recall"], color="k", ls="--",
                   lw=0.8, label="deployed rule")
    axs[0].set(xscale="log", xlabel="false-alarm budget (per hour per class)", ylabel="recall, test streams (%)")
    axs[1].set(xscale="log", xlabel="false-alarm budget (per hour per class)", ylabel="alerts per hour,\nreal home (SINS)")
    for ax in axs:
        ax.legend(fontsize=6)
    save(fig, "fig_transfer")


def real_scenes():
    """Naturally recorded audio: gate wake-up per DESED event class, classifier duty per SINS activity."""
    p = os.path.join(C.RESULTS, "final.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))[FINAL]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.4))
    w = sorted(r["desed"]["wake_recall"].items(), key=lambda kv: kv[1]["rate"])
    axs[0].barh([k.replace("_", " ").replace(" toothbrush", "") for k, _ in w], [100 * v["rate"] for _, v in w], color=COL["F3"])
    axs[0].set(xlim=(80, 100), xlabel="gate awake during labelled event (%)", title=f"DESED, {r['desed']['clips']} real clips")
    a = sorted(r["sins"].items(), key=lambda kv: kv[1]["duty"])
    axs[1].barh([k.replace("_", " ") for k, _ in a], [100 * v["duty"] for _, v in a], color=COL["F4"])
    axs[1].axvline(100 * r["sins_all"]["duty"], color="k", ls="--", lw=0.7)
    axs[1].set(xlim=(0, 100), xlabel="classifier duty (%)", title=f"SINS, {r['sins_all']['hours']:.1f} h of home life")
    for ax in axs:
        ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig_real_scenes")


def example(r):
    fe = r["frontends"].get("F4@1%", {}).get("frontend")
    if not os.path.isdir(os.path.join(C.DATA, "demand")):
        return
    spec = next(s for s in streams.specs() if s["split"] == "test" and s["env"] == "DLIVING" and s["kind"] == "events")
    x, events = streams.render(spec)
    level, bands = frontend.band_power(x)
    n = (len(x) - C.WIN) // C.HOP + 1
    fig, axs = plt.subplots(3, 1, figsize=(7, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 0.5]))
    f, t, S = signal.spectrogram(x, C.SR, nperseg=512, noverlap=256)
    axs[0].pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-40,
                      rasterized=True)
    axs[0].set_ylabel("kHz")
    for e in events:
        if e["role"] == "target":
            axs[0].axvspan(e["onset"], e["end"], color="w", alpha=0.15)
            axs[0].text(e["onset"], 7.3, f"{e['category'].lower()} {e['snr']:.0f} dB", color="w", fontsize=5)
    if fe:
        stat = frontend.gate_novelty(bands, n, C.GATE_VARIANTS[fe["variant"]])
        tw = (C.WIN + np.arange(n) * C.HOP) / C.SR
        axs[1].plot(tw, stat, lw=0.6, color=COL["F3"])
        axs[1].axhline(fe["thr"], color="k", ls="--", lw=0.7)
        awake = E.hang(stat > fe["thr"], fe["hang"])
        axs[2].fill_between(tw, 0, awake, step="pre", color=COL["F4"], alpha=0.8)
        axs[2].set_title(f"classifier awake {100 * awake.mean():.0f} % of the time", fontsize=7)
        axs[1].set_ylabel(f"gate statistic\n({fe['variant']}, dB)", fontsize=6)
    axs[2].set(yticks=[], xlabel="time (s)")
    save(fig, "fig_example")


def main():
    r = json.load(open(os.path.join(C.RESULTS, "results.json")))
    filters()
    pareto(r)
    idle_duty(r)
    curve()
    transfer()
    alarm_character()
    real_scenes()
    example(r)
    print("  figures ->", C.FIGURES)


if __name__ == "__main__":
    main()
