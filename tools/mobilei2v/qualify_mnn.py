#!/usr/bin/env python3
"""Qualify one original FP32 VAE ONNX -> MNN stage on the host CPU only.

--source-dir accepts the extracted stage directory or a root containing STAGE/;
--vae-stage-dir accepts only the direct stage directory. Both require the exact
local qualify_vae.py. --output-dir/STAGE must be fresh (a previous
report alone is allowed). Supply the parent's pinned MNNConvert and compiled
mnn_forward with --converter/--runner/--mnn-source. Nothing is downloaded.
The runner interface is: mnn_forward --manifest /absolute/path/manifest.json.
This diagnostic cannot qualify an Android/Adreno pack or end-to-end MobileI2V.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import qualify_vae as vae

MNN_COMMIT = "3db3cc904dfea55286972b472b040ad5525aa083"
MOBILEI2V_COMMIT = "8d0a253c766b05a43ba408baf5e8f800a36be8b4"
ATOL, RTOL = 1e-4, 1e-3
UPSTREAM_LOCK = Path(__file__).with_name("upstream.lock.properties")
MAX_BYTES = (1 << 31) - 1


def checked_count(shape):
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("shape must have positive static integer dimensions")
    count = math.prod(shape)
    if count > MAX_BYTES // 4:
        raise ValueError("tensor exceeds bounded float32 byte count")
    return count


def regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular non-symlink file required: {path}")
    return path


def verify_identity(path, expected, name):
    regular_file(path)
    if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int or expected["bytes"] <= 0
            or Path(path).stat().st_size != expected["bytes"]
            or vae.file_identity(path) != expected):
        raise ValueError(f"{name} artifact identity changed: {path}")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    regular_file(path)
    if Path(path).stat().st_size > 4 * 1024 * 1024:
        raise ValueError("report exceeds bounded JSON size")
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("JSON report must be an object")
    return result


def validate_source_report(report, stage, directory):
    contract = vae.expected_contract(stage)
    lock = vae.read_lock()
    upstream = {}
    for line in UPSTREAM_LOCK.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() in upstream:
                raise ValueError("duplicate upstream lock key")
            upstream[key.strip()] = value.strip()
    if (upstream.get("source.commit") != MOBILEI2V_COMMIT
            or upstream.get("source.repo") != "hustvl/MobileI2V"):
        raise ValueError("MobileI2V source lock mismatch")
    required = {
        "format": "mobilei2v-baseline-vae-qualification-v1", "stage": stage,
        "phase": "ort", "execution": "host-cpu", "mobilei2v_source_commit": MOBILEI2V_COMMIT,
        "boundary_dtype": "float32", "weight_dtype": "float32", "operation_dtype": "float32",
        "scaling_factor": 1.0, "opset": 17, "onnx_opsets": {"ai.onnx": 17},
        "expected_contract": contract, "onnx_contract": contract,
        "pins": {k: v for k, v in lock.items() if k.startswith("vae.")},
        "checkpoint": {"bytes": int(lock["vae.weights.bytes"]), "sha256": lock["vae.weights.sha256"]},
        "auxiliary_lock": vae.file_identity(vae.LOCK_PATH), "script": vae.file_identity(vae.__file__),
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"source report {key} mismatch")
    for key in ("vae_qualification_passed", "export_passed", "onnx_checker_passed", "pytorch_full_shape_forward"):
        if report.get(key) is not True:
            raise ValueError(f"source report requires {key}=true")
    for key in ("android_gpu_pack_ready", "reference_tiling", "latents_mean_std_normalization"):
        if report.get(key) is not False:
            raise ValueError(f"source report requires {key}=false")
    if "error" in report:
        raise ValueError("source report contains an error")
    parity = report.get("onnx_parity", {})
    if (parity.get("passed") is not True or parity.get("finite") is not True
            or parity.get("atol") != ATOL or parity.get("rtol") != RTOL
            or type(parity.get("elements")) is not int
            or parity["elements"] != checked_count(next(iter(contract["outputs"].values()))["shape"])):
        raise ValueError("source parity must pass with exactly atol=1e-4, rtol=1e-3 and the full output")
    for metric in ("max_absolute_error", "rmse"):
        value = parity.get(metric)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"source parity {metric} is invalid")
    artifacts = report.get("artifacts", {})
    required_names = {"model.onnx", "reference.npy", *[f"{name}.npy" for name in contract["inputs"]]}
    if not isinstance(artifacts, dict) or set(artifacts) != required_names:
        raise ValueError("source artifact set must bind exactly graph, reference, and every input")
    for name in sorted(required_names):
        verify_identity(directory / name, artifacts[name], "source")
    if not 0 < (directory / "model.onnx").stat().st_size < (1 << 31):
        raise ValueError("only the original embedded ONNX branch below 2GB is supported")
    return contract


def graph_contract(graph):
    contract = {}
    for group, values in (("inputs", graph.input), ("outputs", graph.output)):
        ports = {}
        for value in values:
            if not value.name or value.name in ports or not value.type.HasField("tensor_type"):
                raise ValueError(f"duplicate, unnamed, or non-tensor ONNX {group} port")
            tensor = value.type.tensor_type
            # TensorProto.FLOAT == 1. Reject HALF before pinned MNN setType,
            # which does not implement DT_HALF and can retain a float default.
            if tensor.elem_type != 1 or not tensor.HasField("shape"):
                raise ValueError(f"ONNX port {value.name} must have a float32 static shape")
            if any(not dim.HasField("dim_value") for dim in tensor.shape.dim):
                raise ValueError(f"dynamic ONNX port shape: {value.name}")
            shape = [dim.dim_value for dim in tensor.shape.dim]
            checked_count(shape)
            ports[value.name] = {"dtype": "float32", "shape": shape}
        contract[group] = ports
    return contract


def inspect_onnx(path, stage):
    import onnx  # Required in parent CI; never install or silently skip.

    model = onnx.load(str(path), load_external_data=False)
    if model.functions or {op.domain or "ai.onnx": op.version for op in model.opset_import} != {"ai.onnx": 17}:
        raise ValueError("only the original ONNX opset 17 graph without local functions is supported")

    def tensor(value):
        if value.data_location == onnx.TensorProto.EXTERNAL or value.external_data:
            raise ValueError("unbound external ONNX tensor data")
        if value.data_type not in (1, 2, 3, 4, 5, 6, 7, 9, 12, 13):
            raise ValueError("ONNX floating constants must be float32 (integer shape constants allowed)")

    def visit(graph):
        for value in graph.initializer:
            tensor(value)
        for value in graph.sparse_initializer:
            tensor(value.values)
            tensor(value.indices)
        for node in graph.node:
            if node.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                                "Bernoulli", "Multinomial"}:
                raise ValueError("implicit random sampling in ONNX graph")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor(attr.t)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for value in attr.tensors:
                        tensor(value)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        visit(child)

    visit(model.graph)
    contract = graph_contract(model.graph)
    vae.validate_graph_contract(stage, contract)
    del model
    gc.collect()
    onnx.checker.check_model(str(path))
    return contract


def load_fixture(path, shape, name):
    regular_file(path)
    count = checked_count(shape)
    if not count * 4 < Path(path).stat().st_size <= count * 4 + 16384:
        raise ValueError(f"{name} NPY byte count is incompatible with float32 shape")
    value = np.load(path, allow_pickle=False, mmap_mode="r", max_header_size=16384)
    vae.validate_array(value, shape, name)
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must use contiguous C order")
    return value


def write_input_binary(source, destination, shape):
    if sys.byteorder != "little":
        raise ValueError("runner binary fixtures require a little-endian host")
    value = load_fixture(source, shape, "input")
    with Path(destination).open("xb") as handle:
        value.tofile(handle)
    if Path(destination).stat().st_size != checked_count(shape) * 4:
        raise ValueError("input binary byte count mismatch")


def parity_metrics(reference, actual):
    result = {"passed": False, "atol": ATOL, "rtol": RTOL,
              "reference_shape": list(reference.shape), "actual_shape": list(actual.shape),
              "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype),
              "finite": False, "max_absolute_error": None, "rmse": None}
    try:
        result.update(vae.compare_outputs(reference, actual, atol=ATOL, rtol=RTOL))
    except ValueError as error:
        result["error"] = str(error)
    return result


def compare_binary(reference_path, output_path, shape):
    result = {"passed": False, "atol": ATOL, "rtol": RTOL, "expected_shape": list(shape),
              "finite": False, "max_absolute_error": None, "rmse": None}
    try:
        reference = load_fixture(reference_path, shape, "PyTorch reference")
        regular_file(output_path)
        result["expected_bytes"] = checked_count(shape) * 4
        result["actual_bytes"] = Path(output_path).stat().st_size
        if result["actual_bytes"] != result["expected_bytes"]:
            raise ValueError("MNN output byte count/shape mismatch")
        actual = np.memmap(output_path, dtype="<f4", mode="r", shape=tuple(shape), order="C")
        result.update(parity_metrics(reference, actual))
    except (ValueError, OSError) as error:
        result["error"] = str(error)
    return result


def git_text(source, *arguments):
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(source), *arguments],
                            check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def verify_mnn_source(source):
    source = Path(source).resolve()
    if Path(git_text(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("--mnn-source must be the MNN checkout root")
    commit = git_text(source, "rev-parse", "HEAD")
    if commit != MNN_COMMIT:
        raise ValueError("MNN source commit differs from the pin")
    if git_text(source, "status", "--porcelain=v1", "--untracked-files=normal", "--ignore-submodules=none"):
        raise ValueError("MNN source tree is dirty")
    return {"path": str(source), "commit": commit, "clean": True}


def write_report(directory, report):
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_command(command, name, args, directory, report):
    record = {"argv": command, "timeout_seconds": args.timeout_seconds, "returncode": None}
    report.setdefault("commands", {})[name] = record
    write_report(directory, report)
    started = time.monotonic()
    try:
        with (directory / f"{name}.log").open("xb") as log:
            result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout_seconds, check=False)
        record["returncode"] = result.returncode
        if result.returncode != 0:
            raise ValueError(f"{name} failed (exit {result.returncode}); see {name}.log")
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        log_path = directory / f"{name}.log"
        if log_path.is_file():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 16384))
                record["log_tail"] = log.read(16384).decode("utf-8", errors="replace")


def qualify(args, directory, report):
    source = args.vae_stage_dir.resolve()
    report_path = source / "report.json"
    source_identity = vae.file_identity(regular_file(report_path))
    source_report = read_json(report_path)
    contract = validate_source_report(source_report, args.stage, source)
    report["source_report"] = {"path": str(report_path), **source_identity}
    report["source_artifacts"] = source_report["artifacts"]
    report["source_onnx_parity"] = source_report["onnx_parity"]
    report["checkpoint"] = source_report["checkpoint"]
    report["pins"] = source_report["pins"]
    report["mobilei2v_source_commit"] = MOBILEI2V_COMMIT
    bindings = {Path(vae.__file__).resolve(): source_report["script"],
                vae.LOCK_PATH.resolve(): source_report["auxiliary_lock"],
                UPSTREAM_LOCK.resolve(): vae.file_identity(UPSTREAM_LOCK),
                Path(__file__).resolve(): vae.file_identity(__file__),
                Path(__file__).with_name("mnn_forward.cpp").resolve():
                    vae.file_identity(Path(__file__).with_name("mnn_forward.cpp"))}
    report["local_code_and_locks"] = {str(path): ident for path, ident in bindings.items()}
    report["mnn_source"] = verify_mnn_source(args.mnn_source)
    for name in ("converter", "runner"):
        path = regular_file(getattr(args, name).resolve())
        if not os.access(path, os.X_OK) or path.stat().st_size == 0:
            raise ValueError(f"{name} must be a nonempty executable")
        bindings[path] = vae.file_identity(path)
        report[name] = {"path": str(path), **bindings[path]}
    report["onnx_contract"] = inspect_onnx(source / "model.onnx", args.stage)
    output_name, output_spec = next(iter(contract["outputs"].items()))
    # Validate reference and all inputs before spending converter/runtime work.
    load_fixture(source / "reference.npy", output_spec["shape"], "PyTorch reference")
    manifest = {"format": "mobilei2v-mnn-forward-v1", "stage": args.stage, "byte_order": "little",
                "model": str(directory / "model.mnn"), "threads": args.threads, "inputs": {}, "outputs": {}}
    artifacts = report.setdefault("artifacts", {})
    for name, spec in contract["inputs"].items():
        path = directory / f"{name}.bin"
        write_input_binary(source / f"{name}.npy", path, spec["shape"])
        artifacts[path.name] = vae.file_identity(path)
        manifest["inputs"][name] = {**spec, "path": str(path)}
    output_path = directory / f"{output_name}.bin"
    manifest["outputs"][output_name] = {**output_spec, "path": str(output_path)}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    artifacts[manifest_path.name] = vae.file_identity(manifest_path)
    # No --fp16, weight quantization, custom ops, benchmark stripping, or graph rewrite.
    run_command([report["converter"]["path"], "-f", "ONNX", "--modelFile", str(source / "model.onnx"),
                 "--MNNModel", manifest["model"], "--bizCode", "MobileI2V", "--keepInputFormat=1"],
                "convert", args, directory, report)
    model_path = regular_file(directory / "model.mnn")
    if not 0 < model_path.stat().st_size < (1 << 31):
        raise ValueError("MNN model must be nonempty and below 2GB")
    # Embedded original branches only; reject converter-created weight sidecars.
    expected_files = {"report.json", "manifest.json", "convert.log", "model.mnn", *artifacts}
    if {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError("unexpected conversion artifacts (external weights/subgraphs are not supported)")
    artifacts[model_path.name] = vae.file_identity(model_path)
    report["conversion_passed"] = True
    for name, ident in artifacts.items():
        verify_identity(directory / name, ident, "runner input")
    run_command([report["runner"]["path"], "--manifest", str(manifest_path)],
                "runner", args, directory, report)
    report["mnn_parity"] = compare_binary(source / "reference.npy", output_path, output_spec["shape"])
    if output_path.is_file() and not output_path.is_symlink():
        artifacts[output_path.name] = vae.file_identity(output_path)
    write_report(directory, report)  # Keep numerical/shape/nonfinite metrics on failure.
    if not report["mnn_parity"]["passed"]:
        raise ValueError("MNN/PyTorch VAE parity failed; see mnn_parity")
    verify_identity(report_path, source_identity, "source report")
    for name, ident in source_report["artifacts"].items():
        verify_identity(source / name, ident, "source")
    for path, ident in bindings.items():
        verify_identity(path, ident, "executable/code/lock")
    for name, ident in artifacts.items():
        verify_identity(directory / name, ident, "MNN")
    report["mnn_source_after"] = verify_mnn_source(args.mnn_source)
    report["identities_rechecked_after_run"] = True
    report["mnn_host_parity_passed"] = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("encoder", "decoder"), required=True)
    source_args = parser.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--vae-stage-dir", type=Path, help="direct qualified stage directory")
    source_args.add_argument("--source-dir", type=Path, help="direct stage directory or root containing STAGE/")
    parser.add_argument("--output-dir", type=Path, required=True, help="writes to OUTPUT_DIR/STAGE")
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--mnn-source", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="per conversion/runner subprocess")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16 or not 1 <= args.timeout_seconds <= 86400:
        parser.error("threads must be 1..16 and timeout-seconds must be 1..86400")
    if args.source_dir is not None:
        base = args.source_dir.resolve()
        if (base / "report.json").exists() and (base / args.stage / "report.json").exists():
            parser.error("ambiguous --source-dir: both direct and stage-nested reports exist")
        args.vae_stage_dir = base / args.stage if (base / args.stage / "report.json").exists() else base
    directory = (args.output_dir / args.stage).resolve()
    source = args.vae_stage_dir.resolve()
    if directory == source or source in directory.parents or directory in source.parents:
        parser.error("source stage and output stage directories must not overlap")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "report.json").is_symlink():
        parser.error("output report must not be a symlink")
    report = {"format": "mobilei2v-baseline-mnn-host-qualification-v1", "stage": args.stage,
              "execution": "host-cpu", "backend": "MNN_FORWARD_CPU", "precision": "Precision_High",
              "runtime_hints": {"WINOGRAD_MEMORY_LEVEL": 0},
              "qualification_scope": "one VAE stage; CPU diagnostic only; not end-to-end I2V or Adreno",
              "cpu_diagnostic_only": True, "android_gpu_pack_ready": False,
              "conversion_passed": False, "mnn_host_parity_passed": False,
              "mnn_parity": {"passed": False, "atol": ATOL, "rtol": RTOL, "error": "not executed"},
              "threads": args.threads, "host": {"system": platform.system(), "machine": platform.machine()},
              "remaining": ["other VAE stage and end-to-end MobileI2V acceptance",
                            "actual handset Adreno execution, correctness, and performance"]}
    started, status = time.monotonic(), 1
    write_report(directory, report)  # Invalidate a stale pass before any expensive operation.
    try:
        if {p.name for p in directory.iterdir()} != {"report.json"}:
            raise ValueError("output stage must be fresh; use a new --output-dir to prevent stale outputs")
        qualify(args, directory, report)
        status = 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        if status:
            report["mnn_host_parity_passed"] = False
        report["android_gpu_pack_ready"] = False
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(directory, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
