"""Write paper/generated/*.tex from results/results.json and results/stress.json, so every number in
the paper comes from the run (python paper/make_tables.py; also called by run_all.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402

OUT = os.path.join(C.ROOT, "paper", "generated")
ROWS = [("F0", "F0 always on"), ("F1@1%", "F1 every $k$-th"), ("F2@1%", "F2 fixed gate"),
        ("F3 v1@1%", "F3 adaptive, v1"), ("F3@1%", "F3 adaptive"), ("F0+A", "F0 + alarm DSP"),
        ("F4@1%", "\\textbf{F4 proposed}"), ("F4@3%", "F4 (3\\,\\% budget)"), ("F4@5%", "F4 (5\\,\\% budget)"),
        ("F4 strict@1%", "F4, strict T-3")]


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def main():
    r = json.load(open(os.path.join(C.RESULTS, "results.json")))
    os.makedirs(OUT, exist_ok=True)
    f = r["frontends"]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Front end & Recall & Fire & Siren & Door & FA/h & Duty idle & Compute \\",
             r" & (\%) & (\%) & (\%) & (\%) & & (\%) & (rel.) \\", r"\midrule"]
    for k, label in ROWS:
        if k not in f:
            continue
        x = f[k]
        bc = x["by_category"]
        fa = x["fa_per_hour_idle"] * r["hours"]["test_idle"] + x["fa_per_hour_events"] * r["hours"]["test_events"]
        fa /= r["hours"]["test_idle"] + r["hours"]["test_events"]
        lines.append(f"{label} & {pct(x['recall'])} & {pct(bc['FIRE_ALARM']['recall'])} & {pct(bc['SIREN']['recall'])} & "
                     f"{pct(bc['DOOR']['recall'])} & {fa:.1f} & {pct(x['duty_idle'], 0)} & {x['compute_idle']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "main_table.tex"), "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Gate statistic & Tune duty & Test duty idle & Test recall \\",
             r" & (\%) & (\%) & (\%) \\", r"\midrule"]
    for v, a in r["ablation"].items():
        star = "$^\\ast$" if v == r["selected_variant"] else ""
        lines.append(f"{v.replace('+', ' + ')}{star} & {pct(a['tune_duty'])} & {pct(a['test_duty_idle'])} & "
                     f"{pct(a['test_recall'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "ablation_table.tex"), "w").write("\n".join(lines) + "\n")

    p = os.path.join(C.RESULTS, "stress.json")
    if os.path.exists(p):
        s = json.load(open(p))
        lines = [r"\begin{tabular}{lcrrrr}", r"\toprule", r"Signal & Fire? & YAMNet & T-3 & Strict & 2 cycles \\", r"\midrule"]
        for n, x in s.items():
            label = n.replace("x0.7", r"$\times$0.7").replace("x1.5", r"$\times$1.5")
            lines.append(f"{label} & "
                         f"{'yes' if x['expected_fire'] else 'no'} & {pct(x['yamnet'], 0)} & {pct(x['t3_loose'], 0)} & "
                         f"{pct(x['t3_strict'], 0)} & {pct(x['t3_confirmed'], 0)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(OUT, "stress_table.tex"), "w").write("\n".join(lines) + "\n")

    # Key numbers as macros for the text
    m = {}
    for key, k in (("Fzero", "F0"), ("Ffour", "F4@1%"), ("Fthree", "F3@1%"), ("Ftwo", "F2@1%"), ("Fone", "F1@1%")):
        if k in f:
            x = f[k]
            m[f"{key}Recall"] = pct(x["recall"])
            m[f"{key}Fire"] = pct(x["by_category"]["FIRE_ALARM"]["recall"])
            m[f"{key}DutyIdle"] = pct(x["duty_idle"], 0)
            m[f"{key}Compute"] = f"{x['compute_idle']:.2f}"
            m[f"{key}Saving"] = f"{100 * (1 - x['compute_idle']):.0f}"
            m[f"{key}Lat"] = f"{x['latency_median_s']:.1f}" if x["latency_median_s"] else "--"
    t = r["tests"].get("F4@1% vs F0", {}).get("FIRE_ALARM")
    if t:
        m["FourVsZeroP"] = f"{t['p']:.1e}"
    ops = r["ops_per_window"]
    m["GateOps"] = f"{ops['gate']:,}".replace(",", "\\,")
    m["AlarmOps"] = f"{ops['alarm']:,}".replace(",", "\\,")
    m["DspShare"] = f"{100 * (ops['gate'] + ops['alarm']) / ops['yamnet']:.2f}"
    m["NTest"] = str(r["n_test_targets"])
    m["Variant"] = r["selected_variant"]
    open(os.path.join(OUT, "numbers.tex"), "w").write(
        "".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in m.items()))
    print("  tables ->", OUT)


if __name__ == "__main__":
    main()
