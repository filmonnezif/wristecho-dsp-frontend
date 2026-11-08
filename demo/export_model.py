"""Export the paper's final classifier for the MATLAB demo (run in Colab after eval/improve.py):

  demo/model/mn10_frontend.onnx  EfficientAT mn10 with its log-mel front end written as plain
                                 convolutions (ONNX opset 13, which MATLAB imports). Input: one
                                 0.975 s window at 32 kHz, pre-emphasised (y[n] = x[n+1] - 0.97 x[n])
                                 and reflect-padded by 512 samples on both sides, shape (1, 1, 32223).
                                 Outputs: 527 AudioSet logits and the 960-d embedding.
  demo/model/final_system.mat    the trained head, the category label indices, the per-category score
                                 choice and smoothing, thresholds, gate and knock settings (budget
                                 FINAL_BUDGET false alarms per hour per category), and a test window with
                                 the expected outputs so MATLAB can check the import.
The export is checked against the original model on that test window.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config as C  # noqa: E402
from classifier import efficientat, head  # noqa: E402

NAME, SYSTEM, FINAL_BUDGET = "mn10_as", "best_mn10", "0.1"
OUT = os.path.join(C.ROOT, "demo", "model")


def main():
    import torch
    import torchaudio
    from scipy.io import savemat
    model, mel, _, labels = efficientat._load(NAME)
    model, mel = model.cpu().eval(), mel.cpu().eval()
    n_fft, hop = int(mel.n_fft), int(mel.hopsize)
    win = mel.window.detach().cpu().numpy().astype(np.float64)
    w = np.zeros(n_fft)
    w[(n_fft - len(win)) // 2:(n_fft - len(win)) // 2 + len(win)] = win
    k, f = np.arange(n_fft), np.arange(n_fft // 2 + 1)[:, None]
    kern = np.concatenate([np.cos(2 * np.pi * f * k / n_fft) * w, -np.sin(2 * np.pi * f * k / n_fft) * w])
    banks, _ = torchaudio.compliance.kaldi.get_mel_banks(mel.n_mels, n_fft, mel.sr, mel.fmin, mel.fmax, vtln_low=100.0,
                                                        vtln_high=-500.0, vtln_warp_factor=1.0)
    banks = torch.nn.functional.pad(banks, (0, 1)).float()

    class Front(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stft = torch.nn.Conv1d(1, kern.shape[0], n_fft, stride=hop, bias=False)
            self.stft.weight.data = torch.tensor(kern[:, None, :], dtype=torch.float32)
            self.register_buffer("banks", banks)
            self.net = model

        def forward(self, xpad):
            z = self.stft(xpad)
            h = z.shape[1] // 2
            p = z[:, :h] ** 2 + z[:, h:] ** 2
            m = (torch.log(torch.matmul(self.banks, p) + 1e-5) + 4.5) / 5
            logits, feat = self.net(m.unsqueeze(1))
            return logits, feat.reshape(1, -1)

    def prep(x32):
        y = x32[1:] - 0.97 * x32[:-1]
        return np.pad(y, n_fft // 2, mode="reflect").astype(np.float32)

    rng = np.random.default_rng(0)
    t = np.arange(2 * C.WIN) / 32000
    x32 = (0.1 * np.sin(2 * np.pi * 3100 * t) * (t % 0.5 < 0.25) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    front = Front().eval()
    with torch.no_grad():
        ref_logits, ref_feat = model(mel(torch.from_numpy(x32)[None]).unsqueeze(1))
        new_logits, new_feat = front(torch.from_numpy(prep(x32))[None, None])
    err = float((ref_logits - new_logits).abs().max()), float((ref_feat.reshape(1, -1) - new_feat).abs().max())
    print(f"  front-end rewrite vs original: max |logit| error {err[0]:.2e}, max |embedding| error {err[1]:.2e}")
    assert err[0] < 1e-2 and err[1] < 1e-2, err
    os.makedirs(OUT, exist_ok=True)
    onnx = os.path.join(OUT, "mn10_frontend.onnx")
    torch.onnx.export(front, torch.from_numpy(prep(x32))[None, None], onnx, opset_version=13, input_names=["xpad"],
                      output_names=["logits", "embedding"], dynamo=False)
    # the final system's settings
    imp = json.load(open(os.path.join(C.RESULTS, "improve.json")))
    pt = imp["curve"][SYSTEM][FINAL_BUDGET]
    choice = imp["choices"][SYSTEM]
    h = head.load(head.path_for(NAME))
    idx = {l: i for i, l in enumerate(labels)}
    cat_idx = [np.array([idx[l] + 1 for l in C.CATEGORIES[c]["labels"] if l in idx], float) for c in C.ORDER]
    fe = pt["full"]["frontend"]
    savemat(os.path.join(OUT, "final_system.mat"), dict(
        categories=np.array(C.ORDER, dtype=object), head_classes=np.array(C.HEAD_CLASSES, dtype=object),
        mu=h["mu"], sd=h["sd"], stack=float(h["stack"]), n_layers=float(len(h["W"])),
        **{f"W{i}": w for i, w in enumerate(h["W"])}, **{f"b{i}": b.reshape(1, -1) for i, b in enumerate(h["b"])},
        cat_label_idx=np.array(cat_idx, dtype=object),
        source=np.array([choice[c].split(":")[1].split("/")[0] for c in C.ORDER], dtype=object),
        smooth=np.array([float(choice[c].split("/")[1]) for c in C.ORDER]),
        thresholds=np.array([pt["thresholds"][c] for c in C.ORDER]), needed=np.array([float(C.CATEGORIES[c]["needed"]) for c in C.ORDER]),
        gate_delta=float(fe["thr"]), gate_hang=float(fe["hang"]),
        knock_kappa=float(fe["knock"]["kappa"]) if fe.get("knock") else np.nan,
        knock_hits=float(fe["knock"]["hits"]) if fe.get("knock") else np.nan,
        knock_agree=float(fe["knock"]["agree"]) if fe.get("knock") else np.nan,
        test_x32=x32[:2 * C.WIN].astype(np.float64), test_logits=new_logits.numpy().astype(np.float64),
        test_embedding=new_feat.numpy().astype(np.float64), budget=float(FINAL_BUDGET),
        recall_test=pt["full"]["recall"], fa_per_hour_test=pt["full"]["fa_per_hour_events"]))
    print(f"  {onnx} ({os.path.getsize(onnx) / 1e6:.1f} MB), final_system.mat: choice {choice}, thresholds {pt['thresholds']}, gate {fe}")


if __name__ == "__main__":
    main()
