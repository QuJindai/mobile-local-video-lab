#!/usr/bin/env python3
"""Measure CPU FP16 rounding boundaries in the original modulation expression.

This controlled operator example establishes a mechanism, not an attribution of
all denoiser error and not model, MNN or device qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch
import qualify_upstream as upstream


def session(nodes, names):
    graph = helper.make_graph(nodes, "original-modulation-rounding",
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1])
                               for name in names],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    upstream.verify_source(args.upstream)
    source_path = args.upstream / "diffusion/model/nets/sana_blocks.py"
    source = source_path.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "t2i_modulate")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    original = namespace["t2i_modulate"]
    arrays = {"x": np.array([16.], np.float16), "scale": np.array([2.**-11], np.float16),
              "shift": np.array([-16.], np.float16)}
    with torch.inference_mode():
        reference = original(*[torch.from_numpy(arrays[name]) for name in ("x", "shift", "scale")]).numpy()
    one = helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array([1.], np.float16)))
    monolithic = session([one, helper.make_node("Add", ["one", "scale"], ["scale_plus_one"]),
                          helper.make_node("Mul", ["x", "scale_plus_one"], ["product"]),
                          helper.make_node("Add", ["product", "shift"], ["output"])],
                         ["x", "scale", "shift"])
    actual = monolithic.run(["output"], arrays)[0]
    add_scale = session([one, helper.make_node("Add", ["one", "scale"], ["output"])], ["scale"])
    mul = session([helper.make_node("Mul", ["x", "scale"], ["output"])], ["x", "scale"])
    add_shift = session([helper.make_node("Add", ["x", "shift"], ["output"])], ["x", "shift"])
    scale = add_scale.run(["output"], {"scale": arrays["scale"]})[0]
    product = mul.run(["output"], {"x": arrays["x"], "scale": scale})[0]
    split = add_shift.run(["output"], {"x": product, "shift": arrays["shift"]})[0]
    values = (reference, actual, scale, product, split)
    if any(value.shape != (1,) or value.dtype != np.float16 or not np.isfinite(value).all()
           for value in values):
        raise ValueError("rounding control has invalid shape, type or finite status")
    report = {"format": "mobilei2v-fp16-rounding-control-v1", "source_commit": upstream.SOURCE_COMMIT,
              "source_file_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "original_function": ast.get_source_segment(source, node),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "onnxruntime": ort.__version__,
              "execution": "CPUExecutionProvider", "optimization": "ORT_ENABLE_BASIC",
              "inputs": {name: value.tolist() for name, value in arrays.items()},
              "original_pytorch": reference.tolist(), "monolithic_ort": actual.tolist(),
              "split_ort_with_half_boundaries": split.tolist(),
              "split_matches_original": bool(np.array_equal(reference, split)),
              "monolithic_matches_original": bool(np.array_equal(reference, actual)),
              "denoiser_error_attributed": False, "android_gpu_pack_ready": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
