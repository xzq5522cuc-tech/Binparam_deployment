import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from hybrid_partition_bnnfc import make_partitions


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(torch_outputs, ort_outputs):
    records = []
    for expected, actual in zip(torch_outputs, ort_outputs):
        delta = np.abs(expected.detach().cpu().numpy() - actual)
        records.append({"max_abs": float(delta.max()), "mean_abs": float(delta.mean())})
    return records


def remove_identity_aliases(path):
    """Resolve shared-parameter Identity aliases before Bolt conversion.

    PyTorch emits Identity nodes when two LearnableBias modules share the same
    Parameter. Bolt incorrectly aliases several differently-shaped outputs to
    the final Identity name, producing an empty U8 weight. Connecting consumers
    directly to the original initializer is exactly equivalent.
    """
    graph = onnx.load(str(path))
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }

    def resolve(name):
        visited = set()
        while name in aliases and name not in visited:
            visited.add(name)
            name = aliases[name]
        return name

    kept = []
    for node in graph.graph.node:
        if node.op_type == "Identity":
            continue
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        kept.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(kept)
    for output in graph.graph.output:
        output.name = resolve(output.name)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def export_and_validate(module, inputs, path, input_names, output_names, opset):
    module.eval()
    with torch.no_grad():
        expected = module(*inputs)
    if not isinstance(expected, tuple):
        expected = (expected,)
    torch.onnx.export(
        module, inputs, str(path), opset_version=opset,
        input_names=input_names, output_names=output_names,
        export_params=True, do_constant_folding=True,
        keep_initializers_as_inputs=False, dynamic_axes=None,
    )
    remove_identity_aliases(path)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: value.detach().cpu().numpy() for name, value in zip(input_names, inputs)}
    actual = session.run(output_names, feed)
    errors = compare(expected, actual)
    if max(item["max_abs"] for item in errors) > 3e-5:
        raise RuntimeError(f"ONNX mismatch for {path}: {errors}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "nodes": len(graph.graph.node),
        "operators": sorted({node.op_type for node in graph.graph.node}),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("bnn", "fp32"), default="bnn")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cfg_file", default="configs/binparam_pymaf_res50_mix_best.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument(
        "--bolt_bnn_linear_as_conv", action="store_true",
        help="Export binary regressor Linear layers as Bolt-compatible 1x1 Conv",
    )
    args = parser.parse_args()

    from core.cfgs import cfg
    cfg.merge_from_file(args.cfg_file)
    from core import path_config
    from models_toB.inference_layers import convert_binary_layers
    network = importlib.import_module("models_toB.pymaf_net")

    if args.variant == "fp32":
        from models_toB import pose_resnet
        network.ENCODER_BINARIZE = False
        network.get_resnet_encoder = pose_resnet.get_resnet_encoder
        network.DECODER_BINARIZE_LAYER = "none"
        network.KEEP_LAST_LAYER_FULL_PRECISION = False
        network.BINARY_DECONV = False
        network.UPSAMPLE_METHOD = "transpose"
        checkpoint = args.checkpoint or "/data/users/xzq/pymaf/model_weights_f32.pth"
        output_dir = Path(args.output_dir or "deployment/hybrid_onnx_fp32")
    else:
        checkpoint = args.checkpoint or "/data/users/xzq/pymaf/model_weights.pth"
        output_dir = Path(args.output_dir or "deployment/hybrid_onnx_truebnn")

    source = network.pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=False).eval()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    incompatible = source.load_state_dict(state, strict=False)
    legacy_buffers = {f"regressor.{i}.entropy_reg.bin_centers" for i in range(3)}
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - legacy_buffers:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    if args.variant == "fp32":
        inference_model = source
        feature_model, stages = make_partitions(inference_model, full_precision=True)
    else:
        inference_model = convert_binary_layers(source).eval()
        # The Conv rewrite is an exact deployment form of ExportBiLinear and
        # therefore uses torch.sign directly (including sign(0) == 0).  Do
        # not advertise the epsilon policy used by the older Linear branch.
        deployment_sign_epsilon = 0.0 if args.bolt_bnn_linear_as_conv else 1e-4
        feature_model, stages = make_partitions(
            inference_model, stable_binary=True, sign_epsilon=deployment_sign_epsilon,
            bolt_bnn_linear_as_conv=args.bolt_bnn_linear_as_conv,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260821)
    records = {}
    image = torch.randn(1, 3, 224, 224)
    records["feature_pyramid"] = export_and_validate(
        feature_model, (image,), output_dir / f"feature_pyramid_opset{args.opset}.onnx",
        ["image"], ["feature14", "feature28", "feature56"], args.opset,
    )

    pose = inference_model.regressor[0].init_pose.detach().clone()
    shape = inference_model.regressor[0].init_shape.detach().clone()
    camera = inference_model.regressor[0].init_cam.detach().clone()
    for index, points in enumerate((441, 431, 431)):
        raw = torch.randn(1, 256, points)
        inputs = (raw, pose, shape, camera)
        path = output_dir / f"stage{index}_opset{args.opset}.onnx"
        records[f"stage{index}"] = export_and_validate(
            stages[index], inputs, path,
            ["raw_point_feature", "previous_pose", "previous_shape", "previous_camera"],
            ["pred_pose", "pred_shape", "pred_camera"], args.opset,
        )
        with torch.no_grad():
            pose, shape, camera = stages[index](*inputs)

    manifest = {
        "variant": args.variant,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "opset": args.opset,
        "batch": 1,
        "stable_sign_epsilon": (
            deployment_sign_epsilon if args.variant == "bnn" else None
        ),
        "bolt_bnn_linear_as_conv": (
            args.bolt_bnn_linear_as_conv if args.variant == "bnn" else False
        ),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
