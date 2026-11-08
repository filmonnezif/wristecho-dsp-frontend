"""Make the exported mn10 model (demo/export_model.py) importable by MATLAB, and check it with onnxruntime:
  1. EfficientAT's squeeze-excitation blocks become GlobalAveragePool + 1x1 convolutions (MATLAB's importer
     cannot translate the original Gemm-based blocks), and the single-input Unsqueeze/Concat/ReduceMax no-op is
     removed;
  2. the log-mel front end is cut off: MATLAB computes it (WristEchoDemo>mn10) from model/logmel.mat (window,
     mel filterbank), and model/mn10.onnx takes the 128 x 98 log-mel image.
Usage (in a Python with onnx, onnxruntime, scipy): python demo/onnx_for_matlab.py
"""
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference
from scipy.io import loadmat, savemat

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def main():
    src, se = os.path.join(D, "mn10_frontend.onnx"), os.path.join(D, "mn10_se.onnx")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "se_rewrite.py"), src, se,
                    os.path.join(D, "final_system.mat")], check=True)
    m = onnx.load(se)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    stft = next(n for n in g.node if n.op_type == "Conv" and n.input[0] == g.input[0].name)
    win = init[stft.input[1]][0, 0].astype(np.float64)            # the cos kernel of bin 0 = the padded window
    mm = next(n for n in g.node if n.op_type == "MatMul")
    banks = init[mm.input[0]].astype(np.float64)
    first = next(i for i, n in enumerate(g.node) if n.op_type == "Unsqueeze" and any(
        x == n.input[0] for x in [o for k in g.node if k.op_type == "Div" for o in k.output]))
    cut_out = g.node[first].output[0]
    keep = list(g.node[first + 1:])
    for n in keep:
        n.input[:] = ["mel" if x == cut_out else x for x in n.input]
    del g.node[:]
    g.node.extend(keep)
    del g.input[:]
    g.input.extend([helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, 1, banks.shape[0], 98])])
    used = {x for n in g.node for x in n.input}
    ini = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(ini)
    del g.value_info[:]
    m = shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(D, "mn10.onnx"))
    os.remove(se)
    savemat(os.path.join(D, "logmel.mat"), dict(window=win, banks=banks, n_fft=1024.0, hop=320.0, preemph=0.97))
    M = loadmat(os.path.join(D, "final_system.mat"))
    x = M["test_x32"].ravel()
    y = np.pad(x[1:] - 0.97 * x[:-1], 512, mode="reflect")
    fr = np.stack([y[t * 320:t * 320 + 1024] * win for t in range((len(y) - 1024) // 320 + 1)])
    mel = ((np.log(banks @ (np.abs(np.fft.rfft(fr, 1024, axis=1)) ** 2).T + 1e-5) + 4.5) / 5).astype(np.float32)
    o = ort.InferenceSession(os.path.join(D, "mn10.onnx")).run(None, {"mel": mel[None, None]})
    print(f"  mn10.onnx + numpy log-mel vs the original export: max |logit| error "
          f"{np.abs(o[0].ravel() - M['test_logits'].ravel()).max():.1e}")


if __name__ == "__main__":
    main()
