#!/usr/bin/env python3
"""Qualify the *real*, pinned MobileI2V denoiser before Android integration.

This is a host diagnostic/export, not an Android GPU pack or an I2V demo.
It loads the original architecture and checkpoint, executes a full-size forward
pass, exports its actual live inputs, and compares ONNX Runtime with PyTorch.
No VAE, video quality, MNN or handset success is implied by this report.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

SOURCE_COMMIT = "8d0a253c766b05a43ba408baf5e8f800a36be8b4"
CHECKPOINT_SHA256 = "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d"
CHECKPOINT_BYTES = 1074370038


def verify_checkpoint(path: Path, expected_sha=CHECKPOINT_SHA256,
                      expected_bytes=CHECKPOINT_BYTES):
    if path.stat().st_size != expected_bytes:
        raise ValueError("checkpoint size differs from the upstream lock")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError("checkpoint SHA-256 differs from the upstream lock")
    return digest.hexdigest()


def validate_loaded_keys(missing, unexpected):
    # These are generated positional buffers, never learned weights. The
    # upstream inference explicitly removes pos_embed before loading.
    missing_weights = set(missing) - {"pos_embed", "pos_embed_temporal"}
    if missing_weights or unexpected:
        raise ValueError(f"checkpoint parameters mismatch: missing={sorted(missing_weights)}, "
                         f"unexpected={sorted(unexpected)}")


def compare_outputs(reference, actual, *, atol=0.005, rtol=0.01):
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError(f"parity shape mismatch: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("parity tensors must be finite")
    delta = actual - reference
    result = {"passed": bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
              "max_absolute_error": float(np.max(np.abs(delta))),
              "rmse": float(np.sqrt(np.mean(delta * delta))),
              "atol": atol, "rtol": rtol}
    if not result["passed"]:
        raise ValueError("ONNX/PyTorch parity failed: " + json.dumps(result))
    return result


def expected_denoiser_contract():
    return {
        "latent": {"dtype": "float16", "shape": [2, 128, 3, 23, 40]},
        "timestep": {"dtype": "float16", "shape": [2]},
        "cond_mask": {"dtype": "float32", "shape": [1, 2760]},
        "flow_score": {"dtype": "float16", "shape": [2]},
        "output": {"dtype": "float16", "shape": [2, 128, 3, 23, 40]},
    }


def validate_graph_contract(actual):
    expected = expected_denoiser_contract()
    extras = set(actual) - set(expected)
    if extras:
        raise ValueError(f"unexpected live graph inputs/outputs: {sorted(extras)}")
    for name, contract in expected.items():
        if actual.get(name) != contract:
            raise ValueError(f"graph contract mismatch for {name}: {actual.get(name)}")


def verify_source(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("upstream checkout does not match the source lock")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("upstream tracked source was modified")


def import_original_model(root):
    # Load original modules through namespace packages to avoid executing the
    # training/sampler imports in diffusion/__init__.py. No model definitions,
    # layers, parameters, or forward methods are replaced or patched.
    for package in ("diffusion", "diffusion.model", "diffusion.model.nets", "diffusion.utils"):
        if package in sys.modules:
            raise RuntimeError(f"upstream module already loaded: {package}")
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(root.joinpath(*package.split(".")))]
        namespace.__package__ = package
        sys.modules[package] = namespace
    # Upstream treats an installed Triton package as an available CUDA device.
    # Its optional fastlinear import then calls get_device_name(0). The pinned
    # model uses ordinary LiteLA/GLUMBConv; disable only this optional import on
    # an actual CPU host, leaving every executed model layer unchanged.
    import torch
    if not torch.cuda.is_available():
        import_utils = importlib.import_module("diffusion.utils.import_utils")
        import_utils.is_triton_module_available = lambda: False
    importlib.import_module("diffusion.model.nets.mobiledit")
    return importlib.import_module("diffusion.model.builder")


def run(args, report):
    import torch
    import yaml
    import onnx
    import onnxruntime as ort

    root = args.upstream.resolve()
    verify_source(root)
    report["checkpoint_sha256"] = verify_checkpoint(args.checkpoint)
    report["optional_cuda_kernels_disabled"] = not torch.cuda.is_available()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    os.environ["DISABLE_XFORMERS"] = "1"
    builder = import_original_model(root)
    config = yaml.safe_load((root / "configs/mobilei2v_config/MobileI2V_300M_img512.yaml").read_text())
    model_config, text_config = config["model"], config["text_encoder"]
    model = builder.build_model(
        model_config["model"], use_fp32_attention=model_config["fp32_attention"],
        in_channels=config["vae"]["vae_latent_dim"],
        model_max_length=text_config["model_max_length"],
        caption_channels=text_config["caption_channels"],
        y_norm=text_config["y_norm"], y_norm_scale_factor=text_config["y_norm_scale_factor"],
        pred_sigma=config["scheduler"]["pred_sigma"],
        **{key: model_config[key] for key in
           ("attn_type", "ffn_type", "mlp_ratio", "mlp_acts", "qk_norm", "use_pe", "linear_head_dim")})
    # Deserialization only occurs AFTER exact byte-count and SHA verification.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    state.pop("pos_embed", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    validate_loaded_keys(missing, unexpected)
    del checkpoint, state
    model = model.eval().half()
    gc.collect()
    report["loaded_parameters"] = sum(p.numel() for p in model.parameters())
    report["generated_buffers_not_loaded"] = list(missing)

    class Denoiser(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.register_buffer("unused_prompt", torch.zeros(2, 1, 300, 896, dtype=torch.float16))

        def forward(self, latent, timestep, cond_mask, flow_score):
            return self.original(latent, timestep, latent[:1, :, :1],
                                 self.unused_prompt, cond_mask, flow_score, mask=None)

    wrapper = Denoiser(model).eval()
    cond = torch.zeros(1, 2760)
    cond[:, :920] = 1
    latent = torch.randn(1, 128, 3, 23, 40).half().repeat(2, 1, 1, 1, 1)
    inputs = (latent, torch.tensor([500., 500.]).half(), cond, torch.tensor([2., 2.]).half())
    names = ["latent", "timestep", "cond_mask", "flow_score"]
    with torch.inference_mode():
        reference = wrapper(*inputs).float().cpu().numpy()
        if reference.shape != (2, 128, 3, 23, 40) or not np.isfinite(reference).all():
            raise ValueError("real checkpoint forward output has wrong shape or nonfinite values")
        report["pytorch_full_shape_forward"] = True
        # Only a measured invariance is reported; no prompt support is invented.
        wrapper.unused_prompt.normal_()
        changed_prompt = wrapper(*inputs).float().cpu().numpy()
        report["prompt_perturbation_max_abs"] = float(np.max(np.abs(reference - changed_prompt)))
        if not np.array_equal(reference, changed_prompt):
            raise ValueError("prompt affects this checkpoint; the image-only export contract is invalid")
        wrapper.unused_prompt.zero_()
        del changed_prompt
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        print("Exporting the original denoiser", flush=True)
        torch.onnx.export(wrapper, inputs, str(args.onnx), input_names=names,
                          output_names=["output"], opset_version=17, do_constant_folding=True)
    arrays = {name: value.cpu().numpy() for name, value in zip(names, inputs)}
    del wrapper, model, inputs, latent, cond
    gc.collect()
    graph = onnx.load(str(args.onnx), load_external_data=False)
    onnx.checker.check_model(str(args.onnx))
    contract = {}
    for node in [*graph.graph.input, *graph.graph.output]:
        tensor = node.type.tensor_type
        dtype = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16"}.get(
            tensor.elem_type, onnx.TensorProto.DataType.Name(tensor.elem_type).lower())
        contract[node.name] = {"dtype": dtype,
                               "shape": [d.dim_value for d in tensor.shape.dim]}
    del graph
    validate_graph_contract(contract)
    report["onnx_contract"] = contract
    report["current_android_missing_inputs"] = ["prompt", "text_mask"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    print("Comparing ONNX Runtime with PyTorch", flush=True)
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(["output"], arrays)[0]
    report["onnx_parity"] = compare_outputs(reference, actual)
    report["onnx_sha256"] = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    report["denoiser_qualification_passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("threads must be between 1 and 16")
    report = {"format": "mobilei2v-upstream-qualification-v1", "source_commit": SOURCE_COMMIT,
              "execution": "host-cpu-diagnostic", "denoiser_qualification_passed": False,
              "android_gpu_pack_ready": False,
              "remaining": ["VAE artifact provenance and encode/decode parity", "MNN conversion parity",
                            "actual Adreno execution", "image-to-video quality acceptance"]}
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
