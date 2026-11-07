"""Download everything the study needs into DATA_DIR (default ./datasets), skipping what exists.

  YAMNet       MediaPipe yamnet.tflite (labels embedded)
  DEMAND       17 environments, 16 kHz, only the channels in config.CHANNELS (Zenodo 1227121)
  ESC-50       GitHub archive
  FSD50K       only the clips listed in data/cliplist.json (Hugging Face mirror, one file each)
"""
import io
import json
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402

DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"
ESC_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
FSD_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/{dir}/{id}.wav"


def fetch(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e


def demand(env):
    folder = os.path.join(C.DATA, "demand", env)
    want = [f"ch{c:02d}.wav" for c in C.CHANNELS]
    if all(os.path.exists(os.path.join(folder, w)) for w in want):
        return
    z = zipfile.ZipFile(io.BytesIO(fetch(DEMAND_URL.format(env=env))))
    os.makedirs(folder, exist_ok=True)
    for w in want:
        with open(os.path.join(folder, w), "wb") as f:
            f.write(z.read(f"{env}/{w}"))
    print("DEMAND", env, flush=True)


def fsd(clip):
    path = os.path.join(C.DATA, "fsd50k", "clips", clip["dir"], clip["id"] + ".wav")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = fetch(FSD_URL.format(**clip))
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)


def main():
    os.makedirs(C.DATA, exist_ok=True)
    if not os.path.exists(C.MODEL):
        open(C.MODEL, "wb").write(fetch(C.MODEL_URL))
    if not os.path.exists(os.path.join(C.DATA, "ESC-50-master", "meta", "esc50.csv")):
        zipfile.ZipFile(io.BytesIO(fetch(ESC_URL))).extractall(C.DATA)
        print("ESC-50", flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(demand, C.TUNE_ENVS + C.TEST_ENVS))
    clips = json.load(open(os.path.join(C.ROOT, "data", "cliplist.json")))
    clips = [c for c in clips if c["src"] == "fsd"]
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(fsd, clips))
    print(f"FSD50K {len(clips)} clips", flush=True)


if __name__ == "__main__":
    main()
