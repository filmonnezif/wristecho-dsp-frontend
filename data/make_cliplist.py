"""Build data/cliplist.json: which FSD50K / ESC-50 clips are target events or distractors, and in
which split. Run once by the authors (it needs the FSD50K ground-truth CSVs); the committed JSON
is what the experiment uses, so a reproduction only downloads the clips it lists.

Targets: clips whose labels contain exactly one of Siren / Doorbell / Knock / Vehicle horn
(FSD50K) or the ESC-50 classes siren / door_wood_knock / car_horn / crying_baby; Freesound baby
cries (data/freesound_extra.json) and real fire/smoke alarms (data/freesound.json). Distractors: a
fixed sample of everything else, half of it "hard" (sounds easily confused with a target: bells,
phones, doors, impacts, beeps), plus the Freesound look-alike beeps (microwave, reversing beeper...).
Excluded as ambiguous: FSD50K "Alarm" (may contain smoke alarms), "Crying_and_sobbing" (may be a
baby) and ESC-50 clock_alarm.
Splits: FSD50K dev-train + dev-val -> tune, eval -> test; ESC-50 folds 1-4 -> tune, fold 5 -> test;
Freesound design half -> tune, held-out half -> test (split by uploader). ESC-50 clips whose
Freesound source also appears in FSD50K are dropped (cross-dataset leakage).
Tuning clips are further split into "train" (TRAIN_SHARE, the classifier head learns from these) and
"calib" (thresholds, gate and knock settings are chosen on streams made of these), by a hash of the
clip id (Freesound: of the uploader, so one uploader's recordings stay on one side).
"""
import csv
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402

TARGET_FSD = {"Siren": "SIREN", "Doorbell": "DOOR", "Knock": "DOOR", "Vehicle_horn_and_car_horn_and_honking": "CAR_HORN"}
SUBTYPE_FSD = {"Siren": "siren", "Doorbell": "doorbell", "Knock": "knock", "Vehicle_horn_and_car_horn_and_honking": "horn"}
TARGET_ESC = {"siren": ("SIREN", "siren"), "door_wood_knock": ("DOOR", "knock"), "car_horn": ("CAR_HORN", "horn"),
              "crying_baby": ("BABY_CRY", "baby")}
HARD = ["Ringtone", "Telephone", "Bell", "Chime", "Door", "Buzz", "Microwave_oven", "Car_passing_by", "Traffic_noise_and_roadway_noise",
        "Hammer", "Tap", "Thump_and_thud", "Drawer_open_or_close", "Cupboard_open_or_close", "Clock", "Typing",
        "Dishes_and_pots_and_pans", "Speech", "Music", "Beep_and_bleep", "Whistle"]
EXCLUDE = {"Alarm", "Crying_and_sobbing"}
N_HARD = {"tune": 900, "test": 400}
N_EASY = {"tune": 900, "test": 400}


def main():
    rng = random.Random(C.SEED)
    gt = os.path.join(C.DATA, "fsd50k", "FSD50K.ground_truth")
    rows = []
    for name in ("dev", "eval"):
        for r in csv.DictReader(open(os.path.join(gt, f"{name}.csv"))):
            split = "test" if name == "eval" else "tune"
            rows.append(dict(src="fsd", dir=name, id=r["fname"], split=split, labels=r["labels"].split(",")))
    rng.shuffle(rows)
    clips, hard, easy = [], {"tune": [], "test": []}, {"tune": [], "test": []}
    fsd_ids = set()
    for r in rows:
        fsd_ids.add(r["id"])
        t = [l for l in r["labels"] if l in TARGET_FSD]
        if len(t) == 1:  # FSD50K files Siren and Doorbell under "Alarm", so check targets first
            clips.append(dict(r, role="target", category=TARGET_FSD[t[0]], subtype=SUBTYPE_FSD[t[0]]))
        elif not t and not EXCLUDE & set(r["labels"]):
            (hard if any(h in r["labels"] for h in HARD) else easy)[r["split"]].append(r)
    for split in ("tune", "test"):
        clips += [dict(r, role="distractor", category=None, subtype="hard") for r in hard[split][:N_HARD[split]]]
        clips += [dict(r, role="distractor", category=None, subtype="easy") for r in easy[split][:N_EASY[split]]]
    for r in csv.DictReader(open(os.path.join(C.DATA, "ESC-50-master", "meta", "esc50.csv"))):
        if r["src_file"] in fsd_ids or r["category"] == "clock_alarm":
            continue
        split = "test" if r["fold"] == "5" else "tune"
        base = dict(src="esc", dir="esc", id=r["filename"], split=split, labels=[r["category"]])
        if r["category"] in TARGET_ESC:
            cat, sub = TARGET_ESC[r["category"]]
            clips.append(dict(base, role="target", category=cat, subtype=sub))
        else:
            clips.append(dict(base, role="distractor", category=None, subtype="easy"))
    here = os.path.dirname(os.path.abspath(__file__))
    for f in json.load(open(os.path.join(here, "freesound.json"))) + json.load(open(os.path.join(here, "freesound_extra.json"))):
        base = dict(src="fs", dir="freesound", id=str(f["id"]), user=f["user"], preview=f["preview"],
                    split="tune" if f["split"] == "design" else "test", labels=[])
        if f["group"] == "baby cry":
            clips.append(dict(base, role="target", category="BABY_CRY", subtype="baby"))
        elif f["group"] in ("smoke alarm", "fire alarm"):
            clips.append(dict(base, role="target", category="FIRE_ALARM", subtype="real"))
        else:
            clips.append(dict(base, role="distractor", category=None, subtype="lookalike"))
    for c in clips:
        c.pop("labels")
        if c["split"] == "tune":
            h = zlib.crc32((c.get("user") or c["id"]).encode()) % 1000
            c["pool"] = "train" if h < 1000 * C.TRAIN_SHARE else "calib"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cliplist.json")
    json.dump(clips, open(out, "w"), indent=0)
    from collections import Counter
    for k, v in sorted(Counter((c["split"], c.get("pool", ""), c["role"], c["category"] or c["subtype"]) for c in clips).items()):
        print(k, v)


if __name__ == "__main__":
    main()
