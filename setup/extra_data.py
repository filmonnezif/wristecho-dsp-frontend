"""Extra TRAINING clips for the classifier heads (never used for tuning or testing).

  Freesound  python setup/extra_data.py search   (authors, once; needs the API key; writes
             data/freesound_train.json). Recordings titled as a target sound, from uploaders on the
             design side of the uploader split, excluding every uploader and recording of the
             FSD50K evaluation set and the ESC-50 test fold.
  UrbanSound8K  python setup/extra_data.py        (downloads the dataset and the Freesound previews,
             writes data/us8k_train.json: car-horn and siren excerpts whose source recording is not
             in any test set).
The training streams (data/streams.specs(train=True)) draw from the "train" clips plus these.
"""
import csv
import io
import json
import os
import sys
import tarfile
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402

HERE = os.path.join(C.ROOT, "data")
FS_MANIFEST = os.path.join(HERE, "freesound_train.json")
US_MANIFEST = os.path.join(HERE, "us8k_train.json")
US8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
QUERIES = {  # category: (subtype, queries, the title must contain one of)
    "DOOR": [("knock", ["door knock", "knocking on door", "knock knock", "door knocking", "knocking"], ("knock",)),
             ("doorbell", ["doorbell", "door bell", "ding dong doorbell"], ("doorbell", "door bell", "ding dong", "ding-dong"))],
    "CAR_HORN": [("horn", ["car horn", "horn honk", "honking", "truck horn", "car honk"], ("horn", "honk"))],
    "SIREN": [("siren", ["siren", "ambulance siren", "police siren", "fire truck siren", "fire engine siren"], ("siren",))],
    "FIRE_ALARM": [("real", ["smoke alarm", "smoke detector", "fire alarm", "fire alarm bell"], ("smoke", "fire alarm", "detector"))],
    "BABY_CRY": [("baby", ["baby crying", "baby cry", "infant crying", "newborn crying"], ("baby", "infant", "newborn"))],
}
EXCLUDE = ("music", "song", "remix", "loop", "synth", "toy", "fake", "imitat", "cartoon", "game", "battery", "chirp",
           "test button", "low batt", "laugh", "beat", "drum", "french horn", "trumpet", "brass", "horn section",
           "instrument", "tuba", "trombone", "orchestra", "shofar", "vuvuzela", "alphorn", "hunting horn", "shoe horn")
NOT_CAR = ("train", "fog", "ship", "boat", "ferry", "bike", "bicycle", "clown", "squeeze", "vessel", "locomotive",
           "railway", "tram", "cruise")  # horns that are not road-vehicle horns
PER_SUBTYPE = 300


def test_side():
    """Freesound ids and uploaders that any test clip comes from."""
    meta = os.path.join(C.DATA, "fsd50k", "meta", "FSD50K.metadata", "eval_clips_info_FSD50K.json")
    ev = json.load(open(meta))
    ids = {int(k) for k in ev}
    users = {v["uploader"] for v in ev.values()}
    for r in csv.DictReader(open(os.path.join(C.DATA, "ESC-50-master", "meta", "esc50.csv"))):
        if r["fold"] == "5":
            ids.add(int(r["src_file"]))
    for f in json.load(open(os.path.join(HERE, "freesound.json"))) + json.load(open(os.path.join(HERE, "freesound_extra.json"))):
        if f["split"] == "heldout":
            ids.add(int(f["id"]))
            users.add(f["user"])
    return ids, users


def search():
    key = open(os.path.expanduser("~/.config/wristecho/freesound.key")).read().strip()
    bad_ids, bad_users = test_side()
    seen = {int(c["id"]) for c in json.load(open(os.path.join(HERE, "cliplist.json"))) if c["src"] == "fs"}
    out = []
    for cat, groups in QUERIES.items():
        for sub, queries, must in groups:
            n = 0
            for q in queries:
                for page in (1, 2):
                    params = dict(query=q, filter="duration:[1 TO 60]", sort="score", page_size=150, page=page,
                                  fields="id,name,tags,license,username,duration,previews")
                    req = urllib.request.Request("https://freesound.org/apiv2/search/text/?" + urllib.parse.urlencode(params),
                                                 headers={"Authorization": f"Token {key}"})
                    try:
                        res = json.load(urllib.request.urlopen(req, timeout=60))["results"]
                    except Exception:  # noqa: BLE001  (no further page)
                        break
                    for r in res:
                        name = r["name"].lower().replace("_", " ")
                        text = name + " " + " ".join(r["tags"]).lower()
                        if (r["id"] in seen or r["id"] in bad_ids or r["username"] in bad_users or "Sampling" in r["license"]
                                or zlib.crc32(r["username"].encode()) % 2 != 0  # design side of the uploader split only
                                or any(w in text for w in EXCLUDE) or not any(w in name for w in must)
                                or (cat == "CAR_HORN" and any(w in name for w in NOT_CAR))):
                            continue
                        seen.add(r["id"])
                        out.append(dict(src="fs", dir="freesound", id=str(r["id"]), user=r["username"], preview=r["previews"]["preview-hq-mp3"],
                                        split="tune", pool="train", role="target", category=cat, subtype=sub, name=r["name"]))
                        n += 1
                    if n >= PER_SUBTYPE:
                        break
                if n >= PER_SUBTYPE:
                    break
    json.dump(out, open(FS_MANIFEST, "w"), indent=0)
    from collections import Counter
    print(f"  {len(out)} clips -> {FS_MANIFEST}: {Counter((c['category'], c['subtype']) for c in out)}")


def fetch(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def us8k():
    root = os.path.join(C.DATA, "UrbanSound8K")
    if not os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
        tarfile.open(fileobj=io.BytesIO(fetch(US8K_URL)), mode="r:gz").extractall(C.DATA)
    bad_ids, _ = test_side()
    out = []
    for r in csv.DictReader(open(os.path.join(root, "metadata", "UrbanSound8K.csv"))):
        if r["class"] in ("car_horn", "siren") and int(r["fsID"]) not in bad_ids:
            out.append(dict(src="us8k", dir=f"fold{r['fold']}", id=r["slice_file_name"], split="tune", pool="train",
                            role="target", category="CAR_HORN" if r["class"] == "car_horn" else "SIREN",
                            subtype="horn" if r["class"] == "car_horn" else "siren"))
    json.dump(out, open(US_MANIFEST, "w"), indent=0)
    print(f"  UrbanSound8K: {len(out)} training clips")


def previews():
    clips = json.load(open(FS_MANIFEST))
    folder = os.path.join(C.DATA, "freesound")
    os.makedirs(folder, exist_ok=True)

    def get(c):
        p = os.path.join(folder, c["id"] + ".mp3")
        if not os.path.exists(p):
            try:
                data = fetch(c["preview"])
            except Exception:  # noqa: BLE001
                return
            open(p + ".part", "wb").write(data)
            os.replace(p + ".part", p)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, clips))
    print(f"  Freesound training previews: {len(clips)}")


def extra_clips():
    """Extra training clips whose audio is present."""
    from data.streams import clip_path
    out = []
    for p in (FS_MANIFEST, US_MANIFEST):
        if os.path.exists(p):
            out += [c for c in json.load(open(p)) if os.path.exists(clip_path(c))]
    return out


def main():
    previews()
    us8k()


if __name__ == "__main__":
    search() if sys.argv[1:] == ["search"] else main()
