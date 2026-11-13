import argparse
from pathlib import Path

import numpy as np
import onnx


def strict_binary_kind(array):
    values = set(np.unique(array).tolist())
    if values == {-1.0, 1.0}:
        return "BIN11"
    if values == {0.0, 1.0}:
        return "BIN01"
    return None


def audit(path):
    model = onnx.load(str(path))
    initializers = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }
    consumers = {}
    producers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
        for name in node.output:
            producers[name] = node

    totals = {
        "Conv": [0, 0, 0],
        "ConvTranspose": [0, 0, 0],
        "MatMul": [0, 0, 0],
        "Gemm": [0, 0, 0],
    }
    details = []
    sign_edges = []
    for node in model.graph.node:
        if node.op_type == "Sign":
            following = consumers.get(node.output[0], [])
            sign_edges.append((node.name, [(x.name, x.op_type) for x in following]))
        if node.op_type not in totals:
            continue
        arrays = []
        for index, name in enumerate(node.input):
            if name in initializers:
                arrays.append((index, name, initializers[name]))
        # For Conv/Gemm the static weight is normally input 1. For MatMul,
        # either side may be a constant, so report every initializer input.
        for index, name, array in arrays:
            if array.dtype.kind not in "fc":
                continue
            kind = strict_binary_kind(array)
            totals[node.op_type][0] += int(array.size)
            if kind:
                totals[node.op_type][1] += int(array.size)
            eligible = False
            reason = ""
            if node.op_type == "Conv" and array.ndim == 4:
                oc, ic_per_group, kh, kw = array.shape
                group = next((a.i for a in node.attribute if a.name == "group"), 1)
                ic = ic_per_group * group
                eligible = bool(kind and ic % 32 == 0 and oc % 16 == 0 and (kh * kw) % 8 == 1)
                reason = f"ic={ic},oc={oc},k={kh}x{kw}"
            if eligible:
                totals[node.op_type][2] += int(array.size)
            if kind:
                details.append((node.name, node.op_type, index, list(array.shape), kind, eligible, reason))

    print(f"MODEL\t{path.name}\tnodes={len(model.graph.node)}\tbytes={path.stat().st_size}")
    for op, (static_params, binary_params, bolt_conv_eligible) in totals.items():
        print(f"SUMMARY\t{op}\tstatic={static_params}\tbinary={binary_params}\tbolt_bnn_conv_eligible={bolt_conv_eligible}")
    print(f"SIGNS\tcount={len(sign_edges)}")
    for sign, next_nodes in sign_edges:
        print(f"SIGN_EDGE\t{sign}\t{next_nodes}")
    for row in details:
        print("BINARY_WEIGHT\t" + "\t".join(map(str, row)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw in args.paths:
        audit(Path(raw))


if __name__ == "__main__":
    main()
