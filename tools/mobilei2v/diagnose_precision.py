#!/usr/bin/env python3
"""Controlled full-shape denoiser precision experiments; never a GPU pack gate.

Each export/compare phase runs in a fresh process. A numerical failure is a
recorded diagnostic result, not a successful model qualification. Infrastructure,
artifact identity, shape and finite-value failures terminate the experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import qualify_upstream as upstream

PROFILES = ("original-fp16", "promoted-fp32")
NAMES = ("latent", "timestep", "cond_mask", "flow_score")


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def metrics(reference, actual, *, atol=0.005, rtol=0.01):
    reference, actual = np.asarray(reference, np.float32), np.asarray(actual, np.float32)
    if reference.shape != actual.shape or not reference.size:
        raise ValueError("precision comparison shape mismatch")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("precision comparison requires finite tensors")
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowance = atol + rtol * np.abs(reference.astype(np.float64))
    failed = delta > allowance
    worst = int(np.argmax(delta / allowance))
    return {"passed": not bool(failed.any()), "atol": atol, "rtol": rtol,
            "elements": int(reference.size), "failed_elements": int(failed.sum()),
            "max_absolute_error": float(delta.max()),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "worst_tolerance_ratio": float((delta / allowance).flat[worst]),
            "worst_index": [int(index) for index in np.unravel_index(worst, reference.shape)],
            "worst_reference": float(reference.flat[worst]),
            "worst_actual": float(actual.flat[worst])}


def contract(profile):
    if profile not in PROFILES:
        raise ValueError("unknown precision profile")
    result = upstream.expected_denoiser_contract()
    if profile == "promoted-fp32":
        for value in result.values():
            value["dtype"] = "float32"
    return result


def verify_shared_inputs(original, candidate):
    for name in NAMES:
        expected = np.asarray(original[name]).astype(np.float32)
        value = candidate[name]
        if value.dtype != np.float32 or value.shape != expected.shape:
            raise ValueError(f"promoted input dtype/shape mismatch: {name}")
        if not np.isfinite(value).all() or not np.array_equal(expected, value):
            raise ValueError(f"promoted input is not the exact FP16 expansion: {name}")


def load_export(directory, profile):
    report = json.loads((directory / "export.json").read_text())
    required = {"profile": profile, "source_commit": upstream.SOURCE_COMMIT,
                "checkpoint_sha256": upstream.CHECKPOINT_SHA256, "export_completed": True,
                "android_gpu_pack_ready": False, "onnx_contract": contract(profile),
                "script": identity(__file__)}
    for key, value in required.items():
        if report.get(key) != value:
            raise ValueError(f"export report {key} mismatch")
    for name, expected in report["artifacts"].items():
        if Path(name).name != name or identity(directory / name) != expected:
            raise ValueError(f"export artifact identity mismatch: {name}")
    if not {"denoiser.onnx", "fixtures.npz"}.issubset(report["artifacts"]):
        raise ValueError("export must bind graph and exact fixtures")
    return report


def export(args, report):
    import torch
    import yaml
    import onnx

    root = args.upstream.resolve()
    upstream.verify_source(root)
    report["checkpoint_sha256"] = upstream.verify_checkpoint(args.checkpoint)
    if torch.cuda.is_available():
        raise RuntimeError("this controlled experiment requires a CPU host")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    os.environ["DISABLE_XFORMERS"] = "1"
    builder = upstream.import_original_model(root)
    config = yaml.safe_load((root / "configs/mobilei2v_config/MobileI2V_300M_img512.yaml").read_text())
    mc, tc = config["model"], config["text_encoder"]
    model = builder.build_model(
        mc["model"], use_fp32_attention=mc["fp32_attention"],
        in_channels=config["vae"]["vae_latent_dim"], model_max_length=tc["model_max_length"],
        caption_channels=tc["caption_channels"], y_norm=tc["y_norm"],
        y_norm_scale_factor=tc["y_norm_scale_factor"], pred_sigma=config["scheduler"]["pred_sigma"],
        **{key: mc[key] for key in
           ("attn_type", "ffn_type", "mlp_ratio", "mlp_acts", "qk_norm", "use_pe", "linear_head_dim")})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    state.pop("pos_embed", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    upstream.validate_loaded_keys(missing, unexpected)
    del checkpoint, state
    model.eval().half()
    dtype = torch.float16
    if args.profile == "promoted-fp32":
        # Round weights exactly as the upstream FP16 baseline, THEN expand.
        # Loading untouched FP32 checkpoint weights would confound the test.
        model.float()
        dtype = torch.float32
    report.update(loaded_parameters=sum(p.numel() for p in model.parameters()),
                  generated_buffers_not_loaded=list(missing),
                  weight_provenance="checkpoint -> float16" +
                  (" -> exact float32 expansion" if dtype == torch.float32 else ""),
                  arithmetic_profile=args.profile,
                  torch_version=torch.__version__, onnx_version=onnx.__version__)

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = model
            self.register_buffer("unused_prompt", torch.zeros(2, 1, 300, 896, dtype=dtype))
            self.taps = {}

        def forward(self, latent, timestep, cond_mask, flow_score):
            output = self.original(latent, timestep, latent[:1, :, :1],
                                   self.unused_prompt, cond_mask, flow_score, mask=None)
            return (output, *[self.taps[name] for name in tap_names])

    tap_modules = []
    if args.profile == "original-fp16":
        tap_modules = [("x_embedder", model.x_embedder), ("t_block", model.t_block),
                       ("flow_block", model.flow_block)]
        tap_modules += [(f"block_{index}", block) for index, block in enumerate(model.blocks)]
    tap_names = [f"tap_{name}" for name, _ in tap_modules]
    wrapper = Denoiser().eval()
    hooks = []
    for (name, module), tap in zip(tap_modules, tap_names):
        def capture(module, inputs, output, key=tap):
            wrapper.taps[key] = output
        hooks.append(module.register_forward_hook(capture))
    cond = torch.zeros(1, 2760)
    cond[:, :920] = 1
    latent = torch.randn(1, 128, 3, 23, 40).half().repeat(2, 1, 1, 1, 1).to(dtype)
    inputs = (latent, torch.tensor([500., 500.]).half().to(dtype), cond,
              torch.tensor([2., 2.]).half().to(dtype))
    arrays = {name: value.numpy() for name, value in zip(NAMES, inputs)}
    if args.profile == "promoted-fp32":
        load_export(args.directory.parent / "original-fp16", "original-fp16")
        with np.load(args.directory.parent / "original-fp16/fixtures.npz", allow_pickle=False) as original:
            verify_shared_inputs(original, arrays)
        report["exact_shared_input_expansion"] = True
    gc.collect()
    started = time.monotonic()
    try:
        with torch.inference_mode():
            print(f"{args.profile}: original architecture full-shape forward", flush=True)
            outputs = wrapper(*inputs)
            if tuple(outputs[0].shape) != (2, 128, 3, 23, 40):
                raise ValueError("native output shape mismatch")
            for name, value in zip(["reference", *tap_names], outputs):
                array = value.detach().cpu().numpy().copy()
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite native output: {name}")
                arrays[name] = array
            report["native_forward_seconds"] = time.monotonic() - started
            np.savez(args.directory / "fixtures.npz", **arrays)
            del outputs, arrays
            wrapper.taps.clear()
            gc.collect()
            print(f"{args.profile}: export", flush=True)
            torch.onnx.export(wrapper, inputs, str(args.directory / "denoiser.onnx"),
                              input_names=list(NAMES), output_names=["output", *tap_names],
                              opset_version=17, do_constant_folding=True)
    finally:
        for hook in hooks:
            hook.remove()
    wrapper.taps.clear()
    del wrapper, model, inputs, latent, cond
    gc.collect()
    graph = onnx.load(str(args.directory / "denoiser.onnx"))
    taps = []
    for port in list(graph.graph.output)[1:]:
        tensor = port.type.tensor_type
        taps.append({"name": port.name, "dtype": tensor.elem_type,
                     "shape": [dim.dim_value for dim in tensor.shape.dim]})
    del graph.graph.output[1:]
    actual_contract = {}
    for port in [*graph.graph.input, *graph.graph.output]:
        tensor = port.type.tensor_type
        actual_contract[port.name] = {
            "dtype": {1: "float32", 10: "float16"}.get(tensor.elem_type, "unsupported"),
            "shape": [dim.dim_value for dim in tensor.shape.dim]}
    if actual_contract != contract(args.profile):
        raise ValueError(f"unexpected graph contract: {actual_contract}")
    onnx.save(graph, str(args.directory / "denoiser.onnx"))
    del graph
    gc.collect()
    onnx.checker.check_model(str(args.directory / "denoiser.onnx"))
    save(args.directory / "taps.json", taps)
    report.update(export_completed=True, onnx_contract=actual_contract,
                  export_observation="module outputs exposed then pruned" if taps else "output only",
                  artifacts={name: identity(args.directory / name)
                             for name in ("denoiser.onnx", "fixtures.npz", "taps.json")})


def run_ort(path, arrays, threads, outputs):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("unexpected diagnostic provider")
    started = time.monotonic()
    actual = session.run(outputs, arrays)
    elapsed = time.monotonic() - started
    del session
    gc.collect()
    return actual, elapsed


def compare(args, report):
    import onnx
    import onnxruntime as ort
    evidence = load_export(args.directory, args.profile)
    report.update(export_report=identity(args.directory / "export.json"),
                  source_artifacts=evidence["artifacts"], ort_version=ort.__version__,
                  onnx_provider="CPUExecutionProvider", optimization="ORT_ENABLE_BASIC")
    with np.load(args.directory / "fixtures.npz", allow_pickle=False) as fixtures:
        arrays = {name: fixtures[name] for name in NAMES}
        reference = fixtures["reference"]
        actuals, seconds = run_ort(args.directory / "denoiser.onnx", arrays, args.threads, ["output"])
        actual = actuals[0]
        np.save(args.directory / "ort-output.npy", actual, allow_pickle=False)
        report["ort_seconds"] = seconds
        report["onnx_vs_same_precision_pytorch"] = metrics(
            reference, actual, **({"atol": 1e-4, "rtol": 1e-3} if args.profile == "promoted-fp32" else {}))
        save(args.directory / "compare.json", report)
        if args.profile == "promoted-fp32":
            original_dir = args.directory.parent / "original-fp16"
            load_export(original_dir, "original-fp16")
            with np.load(original_dir / "fixtures.npz", allow_pickle=False) as original:
                verify_shared_inputs(original, arrays)
                original_reference = original["reference"]
                report["pytorch_fp32_vs_original_fp16"] = metrics(original_reference, reference)
                report["onnx_fp32_vs_original_fp16"] = metrics(original_reference, actual)
            report["fp32_candidate_passed_all_recorded_comparisons"] = all(
                report[key]["passed"] for key in ("onnx_vs_same_precision_pytorch",
                "pytorch_fp32_vs_original_fp16", "onnx_fp32_vs_original_fp16"))
            report["original_fp16_export_report"] = identity(original_dir / "export.json")
        else:
            graph = onnx.load(str(args.directory / "denoiser.onnx"))
            report["graph_operator_counts"] = dict(Counter(node.op_type for node in graph.graph.node))
            taps = json.loads((args.directory / "taps.json").read_text())
            for tap in taps:
                graph.graph.output.append(onnx.helper.make_tensor_value_info(
                    tap["name"], tap["dtype"], tap["shape"]))
            observed_path = args.directory / "observed-temporary.onnx"
            onnx.save(graph, str(observed_path))
            del graph
            gc.collect()
            try:
                observed, elapsed = run_ort(observed_path, arrays, args.threads,
                                           ["output", *[tap["name"] for tap in taps]])
                report["observed_ort_seconds"] = elapsed
                report["observed_final_vs_pytorch"] = metrics(reference, observed[0])
                report["observation_changed_final"] = not np.array_equal(actual, observed[0])
                report["observed_vs_unobserved_final"] = metrics(actual, observed[0])
                report["observed_layer_comparisons"] = [
                    {"name": tap["name"], **metrics(fixtures[tap["name"]], value)}
                    for tap, value in zip(taps, observed[1:])]
                np.save(args.directory / "observed-output.npy", observed[0], allow_pickle=False)
            finally:
                observed_path.unlink(missing_ok=True)
    report["diagnostic_completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("export", "compare"))
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("threads must be between 1 and 16")
    if args.phase == "export" and (args.upstream is None or args.checkpoint is None):
        parser.error("export requires upstream and checkpoint")
    args.directory = args.output_root / args.profile
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / f"{args.phase}.json"
    if target.exists():
        raise ValueError("refusing to overwrite existing phase evidence")
    report = {"format": "mobilei2v-precision-diagnostic-v1", "phase": args.phase,
              "profile": args.profile, "source_commit": upstream.SOURCE_COMMIT,
              "execution": "host-cpu-diagnostic", "android_gpu_pack_ready": False,
              "denoiser_qualification_passed": False,
              "full_pipeline_qualified": False, "script": identity(__file__)}
    try:
        save(target, report)
        (export if args.phase == "export" else compare)(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(target, report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
