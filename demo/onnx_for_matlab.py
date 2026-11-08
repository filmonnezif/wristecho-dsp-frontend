# Usage: python demo/onnx_for_matlab.py demo/model/mn10_frontend.onnx demo/model/mn10_frontend.onnx demo/model/final_system.mat
"""Rewrite EfficientAT's squeeze-excitation blocks into ops MATLAB's ONNX importer handles natively, and drop
the single-input Unsqueeze/Concat/ReduceMax no-op; check the outputs are unchanged with onnxruntime."""
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from scipy.io import loadmat

src, dst, matf = sys.argv[1:4]
m = onnx.load(src); g = m.graph
init = {i.name: i for i in g.initializer}
nodes = list(g.node)
prod = {o: n for n in nodes for o in n.output}
cons = {}
for n in nodes:
    for x in n.input:
        cons.setdefault(x, []).append(n)
remove, add_before, rename = set(), {}, {}
k = 0
for n in nodes:
    a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
    # SE: ReduceMean(axes 2,3) -> Squeeze -> Squeeze -> Gemm -> Relu -> Gemm -> Sigmoid -> Reshape -> Mul
    if n.op_type == "ReduceMean" and list(a.get("axes", [])) == [2, 3]:
        chain = [n]
        ok = True
        for want in ("Squeeze", "Squeeze", "Gemm", "Relu", "Gemm", "Sigmoid", "Reshape", "Mul"):
            nxt = [c for c in cons.get(chain[-1].output[0], []) if c.op_type == want]
            if len(nxt) != 1:
                ok = False
                break
            chain.append(nxt[0])
        if not ok:
            continue
        rm, sq1, sq2, g1, r, g2, sg, rs, mul = chain
        x = rm.input[0]
        gap = f"se{k}_gap"; c1 = f"se{k}_c1"; r1 = f"se{k}_r1"; c2 = f"se{k}_c2"; s2 = f"se{k}_s"
        new = [helper.make_node("GlobalAveragePool", [x], [gap], name=f"se{k}_n0")]
        for gi, (gemm, out) in enumerate(((g1, c1), (g2, c2))):
            W = numpy_helper.to_array(init[gemm.input[1]]); B = numpy_helper.to_array(init[gemm.input[2]])
            wn, bn = f"se{k}_w{gi}", f"se{k}_b{gi}"
            g.initializer.extend([numpy_helper.from_array(W.reshape(W.shape[0], W.shape[1], 1, 1).astype(np.float32), wn),
                                  numpy_helper.from_array(B.astype(np.float32), bn)])
            new.append(helper.make_node("Conv", [gap if gi == 0 else r1, wn, bn], [out], name=f"se{k}_n{gi + 1}", kernel_shape=[1, 1]))
            if gi == 0:
                new.append(helper.make_node("Relu", [c1], [r1], name=f"se{k}_n1r"))
        new.append(helper.make_node("Sigmoid", [c2], [s2], name=f"se{k}_n3"))
        other = [i for i in mul.input if i != rs.output[0]][0]
        new.append(helper.make_node("Mul", [s2, other], [mul.output[0]], name=f"se{k}_n4"))
        for c in chain:
            remove.add(id(c))
        for c in (sq1, sq2, rs):  # their Constant inputs become unused
            for i in c.input[1:]:
                if i in prod and prod[i].op_type == "Constant":
                    remove.add(id(prod[i]))
        add_before[id(mul)] = new
        k += 1
    # no-op: Unsqueeze(axis 0) -> Concat(single input) -> ReduceMax(axis 0, keepdims 0)
    if n.op_type == "Unsqueeze":
        c = cons.get(n.output[0], [])
        if len(c) == 1 and c[0].op_type == "Concat" and len(c[0].input) == 1:
            r = cons.get(c[0].output[0], [])
            if len(r) == 1 and r[0].op_type == "ReduceMax":
                rename[r[0].output[0]] = n.input[0]
                for z in (n, c[0], r[0]):
                    remove.add(id(z))
                if n.input[1] in prod:
                    remove.add(id(prod[n.input[1]]))
out_nodes = []
for n in nodes:
    if id(n) in add_before:
        out_nodes += add_before[id(n)]
    if id(n) in remove:
        continue
    n.input[:] = [rename.get(i, i) for i in n.input]
    out_nodes.append(n)
del g.node[:]
g.node.extend(out_nodes)
used = {i for n in g.node for i in n.input}
keep = [i for i in g.initializer if i.name in used]
del g.initializer[:]
g.initializer.extend(keep)
del g.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"rewrote {k} SE blocks, removed {len(rename)} no-ops; nodes {len(nodes)} -> {len(g.node)}")
M = loadmat(matf)
x = M["test_x32"].ravel().astype(np.float64)
y = x[1:] - 0.97 * x[:-1]
y = np.pad(y, 512, mode="reflect").astype(np.float32)[None, None]
a = ort.InferenceSession(src).run(None, {"xpad": y})
b = ort.InferenceSession(dst).run(None, {"xpad": y})
print("max |diff| logits", float(np.abs(a[0] - b[0]).max()), "embedding", float(np.abs(a[1] - b[1]).max()),
      "| vs stored test logits", float(np.abs(b[0].ravel() - M["test_logits"].ravel()).max()))
