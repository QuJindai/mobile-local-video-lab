#!/usr/bin/env python3
"""Qualify one original MobileI2V VAE stage on the CPU, using local pinned bytes.

Run encoder and decoder separately. The default --phase all starts sequential
export and ORT subprocesses, releasing PyTorch before ORT loads the graph.
--phase export/ort also allow these steps in separate CI jobs. No downloads,
tiling, dtype reduction, LTX channel normalization, or Android claims are made.
Optional --inputs is an NPZ with the exact named float32 boundary tensors.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

LOCK_PATH = Path(__file__).with_name("auxiliary.lock.properties")
PINS = {
    "vae.repo": "Lightricks/LTX-Video",
    "vae.revision": "8984fa25007f376c1a299016d0957a37a2f797bb",
    "vae.subfolder": "vae",
    "vae.config": "vae/config.json",
    "vae.weights": "vae/diffusion_pytorch_model.safetensors",
    "vae.weights.bytes": "1676798532",
    "vae.weights.sha256": "265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010",
    "vae.scaling_factor": "1.0",
    "vae.posterior": "sample",
    "vae.raw.height": "736",
    "vae.output.crop.rows": "0:720",
}
CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo", "_diffusers_version": "0.32.0.dev0",
    "block_out_channels": [128, 256, 512, 512], "decoder_causal": False,
    "encoder_causal": True, "in_channels": 3, "latent_channels": 128,
    "layers_per_block": [4, 3, 3, 3, 4], "out_channels": 3,
    "patch_size": 4, "patch_size_t": 1, "resnet_norm_eps": 1e-6,
    "scaling_factor": 1.0, "spatio_temporal_scaling": [True, True, True, False],
}
IMAGE_SHAPE = (1, 3, 720, 1280)
GUIDE_SHAPE = (1, 128, 1, 23, 40)
LATENT_SHAPE = (1, 128, 3, 23, 40)
RAW_SHAPE = (1, 3, 17, 736, 1280)
VIDEO_SHAPE = (1, 3, 17, 720, 1280)


def canonical_transpose_perm(perm):
    rank = len(perm)
    if any(axis < -rank or axis >= rank for axis in perm):
        raise ValueError("Transpose axis is outside its tensor rank")
    positive = [axis + rank if axis < 0 else axis for axis in perm]
    if sorted(positive) != list(range(rank)):
        raise ValueError("Transpose axes are not a permutation")
    return positive


def canonicalize_exported_transposes(path):
    # Torch 2.4 symbolic_opset9.movedim leaves negative source axes in ONNX
    # Transpose.perm, while ONNX requires positive indices. Diffusers uses
    # movedim(-1, 1) around RMSNorm. Resolve equivalent axis notation only;
    # learned weights, operations, shapes and arithmetic remain unchanged.
    import onnx
    graph = onnx.load(str(path))
    changes = []

    def visit(current):
        for node in current.node:
            for attribute in node.attribute:
                if node.op_type == "Transpose" and attribute.name == "perm":
                    original = list(attribute.ints)
                    positive = canonical_transpose_perm(original)
                    if original != positive:
                        changes.append({"node": node.name, "before": original, "after": positive})
                        attribute.ints[:] = positive
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(graph.graph)
    if changes:
        onnx.save(graph, str(path))
    return changes


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify_checkpoint(path, expected_sha, expected_bytes):
    if Path(path).stat().st_size != expected_bytes:
        raise ValueError("checkpoint size differs from the auxiliary lock")
    identity = file_identity(path)
    if identity["bytes"] != expected_bytes or identity["sha256"] != expected_sha:
        raise ValueError("checkpoint SHA-256 differs from the auxiliary lock")
    return identity


def read_lock(path=LOCK_PATH):
    lock = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() in lock:
                raise ValueError("duplicate auxiliary lock key")
            lock[key.strip()] = value.strip()
    for key, value in PINS.items():
        if lock.get(key) != value:
            raise ValueError(f"unsupported auxiliary lock pin: {key}")
    return lock


def validate_config(config):
    # This exact snapshot predates optional noisy/timestep-conditioned decoders.
    # Do not silently accept extra config keys that from_config might ignore.
    if config != CONFIG:
        raise ValueError("VAE config differs from the pinned baseline config")


def validate_loaded_keys(missing, unexpected):
    if missing or unexpected:
        raise ValueError(f"weight coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


def require_shape(value, shape, name):
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape must be {shape}, got {tuple(value.shape)}")


def validate_array(value, shape, name):
    require_shape(value, shape, name)
    if value.dtype != np.float32:
        raise ValueError(f"{name} must be float32")
    # Chunk checks keep full-size video diagnostics from allocating full copies.
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1048576):
        if not np.isfinite(flat[start:start + 1048576]).all():
            raise ValueError(f"{name} must be finite")


def sample_posterior_numpy(mean, logvar, epsilon):
    for name, value in (("mean", mean), ("logvar", logvar), ("epsilon", epsilon)):
        validate_array(value, mean.shape, name)
    sample = mean + np.exp(np.float32(0.5) * np.clip(logvar, -30.0, 20.0)) * epsilon
    validate_array(sample, mean.shape, "sample")
    return sample


def crop_video(raw):
    require_shape(raw, RAW_SHAPE, "raw decoder video")
    return raw[:, :, :, :720, :]


def compare_outputs(reference, actual, *, atol=0.0001, rtol=0.001):
    if not all(np.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("parity tolerance must be finite and nonnegative")
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError("parity shape mismatch or empty tensors")
    validate_array(reference, reference.shape, "reference")
    validate_array(actual, reference.shape, "actual")
    ref, got = reference.reshape(-1), actual.reshape(-1)
    maximum, sum_squared, passed = 0.0, 0.0, True
    for start in range(0, ref.size, 1048576):
        left = ref[start:start + 1048576].astype(np.float64)
        right = got[start:start + 1048576].astype(np.float64)
        delta = right - left
        maximum = max(maximum, float(np.abs(delta).max()))
        sum_squared += float(np.sum(delta * delta))
        passed = bool(np.all(np.abs(delta) <= atol + rtol * np.abs(left))) and passed
    return {"passed": passed, "max_absolute_error": maximum,
            "rmse": float(np.sqrt(sum_squared / ref.size)), "atol": atol, "rtol": rtol,
            "elements": int(ref.size), "finite": True}


def expected_contract(stage):
    def tensor(shape):
        return {"dtype": "float32", "shape": list(shape)}
    if stage == "encoder":
        return {"inputs": {"image": tensor(IMAGE_SHAPE), "posterior_epsilon": tensor(GUIDE_SHAPE)},
                "outputs": {"guide": tensor(GUIDE_SHAPE)}}
    if stage == "decoder":
        return {"inputs": {"latent": tensor(LATENT_SHAPE)}, "outputs": {"video": tensor(VIDEO_SHAPE)}}
    raise ValueError("unknown VAE stage")


def validate_graph_contract(stage, contract):
    if contract != expected_contract(stage):
        raise ValueError(f"ONNX {stage} contract mismatch: {contract}")


def versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "torch", "diffusers", "safetensors", "onnx", "onnxruntime"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def make_inputs(args):
    contract = expected_contract(args.stage)["inputs"]
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as archive:
            if set(archive.files) != set(contract):
                raise ValueError(f"input NPZ must contain exactly {list(contract)}")
            arrays = {name: archive[name] for name in contract}
    else:
        rng = np.random.Generator(np.random.PCG64(args.seed))
        arrays = {}
        for name, tensor in contract.items():
            if name == "image":
                arrays[name] = rng.random(tensor["shape"], dtype=np.float32) * 2 - 1
            else:
                arrays[name] = rng.standard_normal(tensor["shape"], dtype=np.float32)
    for name, tensor in contract.items():
        validate_array(arrays[name], tensor["shape"], name)
    if "image" in arrays and (arrays["image"].min() < -1 or arrays["image"].max() > 1):
        raise ValueError("image must be RGB normalized to [-1, 1]")
    return arrays


def load_vae(args, report, lock):
    # Deliberately before importing torch/diffusers/safetensors or deserializing.
    checkpoint = args.vae_dir / Path(lock["vae.weights"]).name
    report["checkpoint"] = verify_checkpoint(checkpoint, lock["vae.weights.sha256"],
                                               int(lock["vae.weights.bytes"]))
    config_path = args.vae_dir / Path(lock["vae.config"]).name
    config = json.loads(config_path.read_text())
    validate_config(config)
    report["config_file"] = file_identity(config_path)
    report["source_config"] = config
    if importlib.metadata.version("diffusers") != "0.35.2":
        raise ValueError("exactly diffusers==0.35.2 is required")

    import torch
    from diffusers import AutoencoderKLLTXVideo
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.normalization import RMSNorm
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Meta construction + strict assignment avoids a second 1.68 GB weight copy.
    with torch.device("meta"):
        vae = AutoencoderKLLTXVideo.from_config(config)
    state = load_file(str(checkpoint), device="cpu")
    expected = vae.state_dict()
    validate_loaded_keys(set(expected) - set(state), set(state) - set(expected))
    manifest = {}
    for name, tensor in state.items():
        require_shape(tensor, expected[name].shape, name)
        if tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} is {tensor.dtype}; FP32 reference required")
        validate_array(tensor.numpy(), tensor.shape, name)
        manifest[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    loaded = vae.load_state_dict(state, strict=True, assign=True)
    validate_loaded_keys(loaded.missing_keys, loaded.unexpected_keys)
    vae.eval().requires_grad_(False)
    if any(t.is_meta or t.dtype != torch.float32 for t in vae.state_dict().values()):
        raise ValueError("VAE weights were not completely materialized as FP32")
    if (vae.spatial_compression_ratio, vae.temporal_compression_ratio) != (32, 8):
        raise ValueError("loaded VAE compression shape is incompatible")
    if any((vae.use_tiling, vae.use_slicing, vae.use_framewise_encoding, vae.use_framewise_decoding)):
        raise ValueError("baseline reference requires untiled VAE execution")
    report["weight_coverage"] = {
        "strict": True, "missing": [], "unexpected": [], "tensors": len(state),
        "elements": sum(t.numel() for t in state.values()),
        "parameters": sum(t.numel() for t in vae.parameters()),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    report["effective_config"] = dict(vae.config)
    report["diffusers_source_files"] = {
        cls.__name__: file_identity(Path(inspect.getfile(cls)))
        for cls in (AutoencoderKLLTXVideo, DiagonalGaussianDistribution, RMSNorm)
    }
    # Coverage is checked for the WHOLE checkpoint before releasing the unused branch.
    del state, expected, tensor
    if args.stage == "encoder":
        del vae.decoder
    else:
        del vae.encoder
    gc.collect()
    return vae


def export_stage(args, directory, report, lock):
    vae = load_vae(args, report, lock)
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, image, posterior_epsilon):
            require_shape(image, IMAGE_SHAPE, "image")
            require_shape(posterior_epsilon, GUIDE_SHAPE, "posterior_epsilon")
            posterior = self.original.encode(image.unsqueeze(2)).latent_dist
            guide = posterior.mean + posterior.std * posterior_epsilon
            require_shape(guide, GUIDE_SHAPE, "guide")
            return guide.reshape(GUIDE_SHAPE)

    class Decoder(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, latent):
            require_shape(latent, LATENT_SHAPE, "latent")
            return crop_video(self.original.decode(latent).sample).reshape(VIDEO_SHAPE)

    arrays = make_inputs(args)
    inputs = tuple(torch.from_numpy(value) for value in arrays.values())
    report["inputs_origin"] = ({"npz": str(args.inputs.resolve()), **file_identity(args.inputs)} if args.inputs else
                               {"synthetic": True, "rng": "numpy.PCG64", "seed": args.seed})
    wrapper = (Encoder(vae) if args.stage == "encoder" else Decoder(vae)).eval()
    with torch.inference_mode():
        if args.stage == "encoder":
            posterior = vae.encode(inputs[0].unsqueeze(2)).latent_dist
            require_shape(posterior.mean, GUIDE_SHAPE, "posterior mean")
            require_shape(posterior.logvar, GUIDE_SHAPE, "posterior logvar")
            reference = (posterior.mean + posterior.std * inputs[1]).numpy()
            numpy_reference = sample_posterior_numpy(posterior.mean.numpy(), posterior.logvar.numpy(),
                                                      arrays["posterior_epsilon"])
            report["numpy_posterior_parity"] = compare_outputs(reference, numpy_reference, atol=1e-6, rtol=1e-5)
            if not report["numpy_posterior_parity"]["passed"]:
                raise ValueError("NumPy/PyTorch explicit posterior sampling parity failed")
            del posterior, numpy_reference
        else:
            raw = vae.decode(inputs[0]).sample
            validate_array(raw.numpy(), RAW_SHAPE, "raw decoder video")
            report["observed_raw_shape"] = list(raw.shape)
            reference = crop_video(raw).contiguous().numpy()
            del raw
        output_shape = GUIDE_SHAPE if args.stage == "encoder" else VIDEO_SHAPE
        validate_array(reference, output_shape, "PyTorch reference")
        report["pytorch_full_shape_forward"] = True
        np.save(directory / "reference.npy", reference, allow_pickle=False)
        del reference
        for name, values in arrays.items():
            np.save(directory / f"{name}.npy", values, allow_pickle=False)
        print(f"Exporting original FP32 VAE {args.stage}", flush=True)
        torch.onnx.export(wrapper, inputs, str(directory / "model.onnx"),
                          input_names=list(arrays), output_names=list(expected_contract(args.stage)["outputs"]),
                          opset_version=17, dynamo=False, do_constant_folding=True)
    del wrapper, vae, inputs, arrays
    gc.collect()
    report["transpose_axis_canonicalization"] = canonicalize_exported_transposes(directory / "model.onnx")
    report["export_passed"] = True
    report["artifacts"] = {
        name: file_identity(directory / name)
        for name in ["model.onnx", "reference.npy", *[f"{n}.npy" for n in expected_contract(args.stage)["inputs"]]]
    }


def ort_stage(args, directory, report):
    if not report.get("export_passed"):
        raise ValueError("successful export report required before ORT qualification")
    for name, identity in report["artifacts"].items():
        if file_identity(directory / name) != identity:
            raise ValueError(f"export artifact changed before ORT parity: {name}")
    required = {"model.onnx", "reference.npy", *[f"{n}.npy" for n in expected_contract(args.stage)["inputs"]]}
    if set(report["artifacts"]) != required:
        raise ValueError("export report must bind the graph, reference, and every input")
    import onnx
    import onnxruntime as ort

    graph_path = directory / "model.onnx"
    graph = onnx.load(str(graph_path), load_external_data=False)
    # Each original branch is below the 2 GB protobuf limit; this exporter emits
    # embedded weights. Refuse unbound sidecars rather than trusting their bytes.
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("unexpected external ONNX weights are not bound by this export report")
    contract = {}
    for group, values in (("inputs", graph.graph.input), ("outputs", graph.graph.output)):
        contract[group] = {}
        for value in values:
            tensor = value.type.tensor_type
            contract[group][value.name] = {
                "dtype": "float32" if tensor.elem_type == onnx.TensorProto.FLOAT else str(tensor.elem_type),
                "shape": [dim.dim_value for dim in tensor.shape.dim],
            }
    validate_graph_contract(args.stage, contract)
    report["onnx_contract"] = contract
    report["onnx_opsets"] = {op.domain or "ai.onnx": op.version for op in graph.opset_import}
    if report["onnx_opsets"].get("ai.onnx") != 17:
        raise ValueError("ONNX opset differs from the export contract")
    random_ops = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial"}
    if any(node.op_type in random_ops for node in graph.graph.node):
        raise ValueError("VAE graph contains implicit random sampling")
    del graph
    gc.collect()
    onnx.checker.check_model(str(graph_path))
    report["onnx_checker_passed"] = True
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False)
              for name in expected_contract(args.stage)["inputs"]}
    for name, value in arrays.items():
        validate_array(value, expected_contract(args.stage)["inputs"][name]["shape"], name)
    reference = np.load(directory / "reference.npy", allow_pickle=False, mmap_mode="r")
    print(f"Comparing original VAE {args.stage}: PyTorch vs ORT CPU", flush=True)
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
    output_name = next(iter(expected_contract(args.stage)["outputs"]))
    actual = session.run([output_name], arrays)[0]
    validate_array(actual, expected_contract(args.stage)["outputs"][output_name]["shape"], "ORT output")
    report["ort_providers"] = session.get_providers()
    report["onnx_parity"] = compare_outputs(reference, actual, atol=args.atol, rtol=args.rtol)
    if not report["onnx_parity"]["passed"]:
        raise ValueError("ONNX/PyTorch VAE parity failed")
    report["vae_qualification_passed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encoder", "decoder"), required=True)
    parser.add_argument("--vae-dir", type=Path, required=True, help="local vae folder with config.json and safetensors")
    parser.add_argument("--output-dir", type=Path, required=True, help="artifacts are written in a stage subdirectory")
    parser.add_argument("--phase", choices=("all", "export", "ort"), default="all")
    parser.add_argument("--inputs", type=Path, help="optional exact float32 input NPZ; export phase only")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.0001)
    parser.add_argument("--rtol", type=float, default=0.001)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16 or args.seed < 0:
        parser.error("threads must be 1..16 and seed must be nonnegative")
    if not all(np.isfinite(t) and t >= 0 for t in (args.atol, args.rtol)):
        parser.error("parity tolerances must be finite and nonnegative")
    directory = args.output_dir / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if args.phase == "all":
        for phase in ("export", "ort"):
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *argv, "--phase", phase])
            if result.returncode:
                print(f"VAE {phase} subprocess failed (exit {result.returncode}); see {report_path}", file=sys.stderr)
                return 1
        return 0
    report = {"format": "mobilei2v-baseline-vae-qualification-v1", "stage": args.stage,
              "mobilei2v_source_commit": "8d0a253c766b05a43ba408baf5e8f800a36be8b4",
              "qualification_scope": "one VAE stage; not end-to-end I2V", "execution": "host-cpu",
              "vae_qualification_passed": False, "export_passed": False, "android_gpu_pack_ready": False,
              "boundary_dtype": "float32", "weight_dtype": "float32", "operation_dtype": "float32",
              "scaling_factor": 1.0, "latents_mean_std_normalization": False,
              "posterior": "mean + exp(0.5 * clamp(logvar, -30, 20)) * explicit posterior_epsilon",
              "image_layout": "NCHW RGB [-1,1]; unsqueeze time axis 2; no external padding",
              "decoder_crop": {"raw_shape": list(RAW_SHAPE), "rows": [0, 720], "resize": False},
              "video_values": "raw decoder RGB float32; no clamp, rescale, uint8 or MP4 encoding",
              "reference_tiling": False, "opset": 17,
              "expected_contract": expected_contract(args.stage),
              "remaining": ["other VAE stage and end-to-end I2V acceptance", "MNN conversion parity",
                            "actual handset Adreno execution and performance"]}
    status = 0
    started = time.monotonic()
    try:
        if args.phase == "ort":
            report = json.loads(report_path.read_text())
            report["vae_qualification_passed"] = False
            report["android_gpu_pack_ready"] = False
            report.pop("error", None)
            if report.get("stage") != args.stage or report.get("expected_contract") != expected_contract(args.stage):
                raise ValueError("export report stage/contract mismatch")
        lock = read_lock()
        lock_identity = file_identity(LOCK_PATH)
        if args.phase == "ort" and report.get("auxiliary_lock") != lock_identity:
            raise ValueError("auxiliary lock changed since export")
        script_identity = file_identity(Path(__file__))
        if args.phase == "ort" and report.get("script") != script_identity:
            raise ValueError("qualification script changed since export")
        report["auxiliary_lock"] = lock_identity
        report["pins"] = {k: v for k, v in lock.items() if k.startswith("vae.")}
        report["script"] = script_identity
        report.setdefault("runtime_versions", {})[args.phase] = versions()
        report.setdefault("threads", {})[args.phase] = args.threads
        report["host"] = {"system": platform.system(), "machine": platform.machine()}
        report["phase"] = args.phase
        # Write a negative report first, so a killed/OOM process cannot leave a stale pass.
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if args.phase == "export":
            export_stage(args, directory, report, lock)
        else:
            ort_stage(args, directory, report)
    except Exception as error:
        report["vae_qualification_passed"] = False
        report["android_gpu_pack_ready"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
        status = 1
    finally:
        report.setdefault("measurements", {})[args.phase] = {
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
