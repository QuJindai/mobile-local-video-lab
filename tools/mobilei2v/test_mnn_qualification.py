"""NumPy-only false-pass tests. No MNN inference or VAE qualification is faked.

ONNX protobuf doubles exercise port validation without installing ONNX locally;
an additional real protobuf check runs when the parent's ONNX is available.
"""
import copy
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import qualify_vae as vae

try:
    import qualify_mnn as qualification
except ModuleNotFoundError as error:
    if error.name != "qualify_mnn":
        raise
    qualification = None


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def source_report(root):
    # Small bytes are ONLY identity-gate fixtures, never accepted as real ONNX.
    for name in ("model.onnx", "image.npy", "posterior_epsilon.npy", "reference.npy"):
        (root / name).write_bytes(b"identity fixture: " + name.encode())
    lock = vae.read_lock()
    return {
        "format": "mobilei2v-baseline-vae-qualification-v1", "stage": "encoder",
        "phase": "ort", "execution": "host-cpu", "export_passed": True,
        "vae_qualification_passed": True, "pytorch_full_shape_forward": True,
        "onnx_checker_passed": True, "android_gpu_pack_ready": False,
        "mobilei2v_source_commit": "8d0a253c766b05a43ba408baf5e8f800a36be8b4",
        "boundary_dtype": "float32", "weight_dtype": "float32", "operation_dtype": "float32",
        "reference_tiling": False, "latents_mean_std_normalization": False,
        "scaling_factor": 1.0, "opset": 17, "onnx_opsets": {"ai.onnx": 17},
        "pins": {k: v for k, v in lock.items() if k.startswith("vae.")},
        "checkpoint": {"bytes": 1676798532,
                       "sha256": "265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010"},
        "auxiliary_lock": identity(vae.LOCK_PATH), "script": identity(vae.__file__),
        "expected_contract": vae.expected_contract("encoder"),
        "onnx_contract": vae.expected_contract("encoder"),
        "onnx_parity": {"passed": True, "finite": True, "atol": 1e-4, "rtol": 1e-3,
                        "max_absolute_error": 0.0, "rmse": 0.0, "elements": 117760},
        "artifacts": {n: identity(root / n) for n in
                      ("model.onnx", "image.npy", "posterior_epsilon.npy", "reference.npy")},
    }


def graph_double():
    def port(name, shape):
        dims = [SimpleNamespace(dim_value=d, HasField=lambda field: field == "dim_value") for d in shape]
        tensor = SimpleNamespace(elem_type=1, shape=SimpleNamespace(dim=dims),
                                 HasField=lambda field: field == "shape")
        return SimpleNamespace(name=name, type=SimpleNamespace(
            tensor_type=tensor, HasField=lambda field: field == "tensor_type"))
    return SimpleNamespace(
        input=[port("image", [1, 3, 720, 1280]), port("posterior_epsilon", [1, 128, 1, 23, 40])],
        output=[port("guide", [1, 128, 1, 23, 40])])


class MNNQualificationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(qualification, "qualify_mnn implementation is missing (RED)")

    def test_source_report_requires_literal_success_and_completed_ort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = source_report(root)
            qualification.validate_source_report(good, "encoder", root)
            for key in ("vae_qualification_passed", "export_passed", "onnx_checker_passed",
                        "pytorch_full_shape_forward"):
                for bad_value in (False, 1, "true", None):
                    bad = copy.deepcopy(good)
                    bad[key] = bad_value
                    with self.subTest(key=key, value=bad_value), self.assertRaises(ValueError):
                        qualification.validate_source_report(bad, "encoder", root)
            for key, value in (("phase", "export"), ("stage", "decoder"),
                               ("android_gpu_pack_ready", True), ("boundary_dtype", "float16"),
                               ("weight_dtype", "float16"), ("operation_dtype", "float16"),
                               ("error", "ORT failed")):
                bad = copy.deepcopy(good)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    qualification.validate_source_report(bad, "encoder", root)

    def test_source_report_binds_checkpoint_commit_script_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = source_report(root)
            for key in ("checkpoint", "auxiliary_lock", "script"):
                bad = copy.deepcopy(good)
                bad[key]["sha256"] = "0" * 64
                with self.subTest(key=key), self.assertRaises(ValueError):
                    qualification.validate_source_report(bad, "encoder", root)
            for key, value in (("mobilei2v_source_commit", "0" * 40), ("pins", {})):
                bad = copy.deepcopy(good)
                bad[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    qualification.validate_source_report(bad, "encoder", root)

    def test_source_report_rejects_relaxed_or_nonfinite_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = source_report(root)
            for key, value in (("passed", False), ("finite", False), ("atol", 0.001),
                               ("rtol", 0.01), ("atol", 0.0), ("rtol", float("nan")),
                               ("rmse", float("inf")), ("elements", 1)):
                bad = copy.deepcopy(good)
                bad["onnx_parity"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    qualification.validate_source_report(bad, "encoder", root)

    def test_changed_graph_or_any_fixture_fails_even_with_same_byte_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = source_report(root)
            for name in good["artifacts"]:
                path = root / name
                original = path.read_bytes()
                path.write_bytes(b"X" + original[1:])
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "artifact"):
                    qualification.validate_source_report(good, "encoder", root)
                path.write_bytes(original)

    def test_artifact_set_must_be_exact_and_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = source_report(root)
            for artifact in ("image.npy", "posterior_epsilon.npy", "reference.npy", "model.onnx"):
                bad = copy.deepcopy(good)
                del bad["artifacts"][artifact]
                with self.subTest(missing=artifact), self.assertRaises(ValueError):
                    qualification.validate_source_report(bad, "encoder", root)
            bad = copy.deepcopy(good)
            bad["artifacts"]["../outside"] = identity(root / "image.npy")
            with self.assertRaises(ValueError):
                qualification.validate_source_report(bad, "encoder", root)
            (root / "image.npy").unlink()
            (root / "image.npy").symlink_to(root / "reference.npy")
            with self.assertRaises(ValueError):
                qualification.validate_source_report(good, "encoder", root)

    def test_actual_ports_reject_half_dynamic_wrong_shape_and_duplicates(self):
        good = graph_double()
        self.assertEqual(qualification.graph_contract(good), vae.expected_contract("encoder"))
        cases = []
        half = copy.deepcopy(good)
        half.input[0].type.tensor_type.elem_type = 10  # TensorProto.FLOAT16
        cases.append(half)
        dynamic = copy.deepcopy(good)
        dynamic.input[0].type.tensor_type.shape.dim[0].HasField = lambda _: False
        cases.append(dynamic)
        wrong_shape = copy.deepcopy(good)
        wrong_shape.output[0].type.tensor_type.shape.dim[-2].dim_value = 22
        cases.append(wrong_shape)
        duplicate = copy.deepcopy(good)
        duplicate.input.append(duplicate.input[0])
        cases.append(duplicate)
        missing = copy.deepcopy(good)
        missing.input.pop()
        cases.append(missing)
        for graph in cases:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                vae.validate_graph_contract("encoder", qualification.graph_contract(graph))

    def test_binary_fixture_uses_exact_native_float32_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([[-1.0, 2.25], [3.0, -0.0]], dtype=np.float32)
            np.save(root / "input.npy", values)
            qualification.write_input_binary(root / "input.npy", root / "input.bin", [2, 2])
            self.assertEqual((root / "input.bin").read_bytes(), values.tobytes(order="C"))
            for bad in (values.astype(np.float16), values.reshape(4), values + np.float32("nan")):
                np.save(root / "input.npy", bad)
                with self.subTest(dtype=bad.dtype, shape=bad.shape), self.assertRaises(ValueError):
                    qualification.write_input_binary(root / "input.npy", root / "bad.bin", [2, 2])

    def test_parity_failure_records_metrics_and_uses_pytorch_relative_tolerance(self):
        reference = np.array([0.0, 100.0], dtype=np.float32)
        actual = np.array([0.25, 100.0], dtype=np.float32)
        result = qualification.parity_metrics(reference, actual)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_absolute_error"], 0.25)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.03125))
        self.assertEqual((result["atol"], result["rtol"]), (1e-4, 1e-3))
        self.assertTrue(qualification.parity_metrics(reference, reference)["passed"])
        # Using the actual tensor in the rtol term would incorrectly pass.
        self.assertFalse(qualification.parity_metrics(
            np.array([1], np.float32), np.array([1.0011005], np.float32))["passed"])

    def test_nonfinite_shape_and_dtype_failures_have_json_safe_metrics(self):
        reference = np.array([0.0, 1.0], dtype=np.float32)
        for actual in (np.array([np.nan, 1], np.float32), np.array([np.inf, 1], np.float32),
                       reference.reshape(1, 2), reference.astype(np.float16), np.array([], np.float32)):
            result = qualification.parity_metrics(reference, actual)
            self.assertFalse(result["passed"])
            self.assertIn("error", result)
            json.dumps(result, allow_nan=False)

    def test_binary_output_rejects_short_extra_and_nonfinite_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = np.array([0, 1], np.float32)
            np.save(root / "reference.npy", reference)
            for content in (b"", reference.tobytes()[:-1], reference.tobytes() + b"x",
                            np.array([0, np.nan], np.float32).tobytes()):
                (root / "output.bin").write_bytes(content)
                result = qualification.compare_binary(root / "reference.npy", root / "output.bin", [2])
                self.assertFalse(result["passed"])
                json.dumps(result, allow_nan=False)

    def test_failed_cli_replaces_stale_pass_without_loading_any_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            report = source_report(source)
            report["vae_qualification_passed"] = False
            original = json.dumps(report)
            (source / "report.json").write_text(original)
            output = root / "out" / "encoder"
            output.mkdir(parents=True)
            (output / "report.json").write_text(json.dumps({"mnn_host_parity_passed": True}))
            result = subprocess.run([
                sys.executable, "-B", qualification.__file__, "--stage", "encoder",
                "--vae-stage-dir", str(source), "--output-dir", str(root / "out"),
                "--converter", "/missing/MNNConvert", "--runner", "/missing/mnn_forward",
                "--mnn-source", "/missing/MNN"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            failed = json.loads((output / "report.json").read_text())
            self.assertFalse(failed["mnn_host_parity_passed"])
            self.assertFalse(failed["android_gpu_pack_ready"])
            self.assertTrue(failed["cpu_diagnostic_only"])
            self.assertIn("vae_qualification_passed", failed["error"])
            self.assertNotIn("No module named", failed["error"])
            self.assertEqual((source / "report.json").read_text(), original)

    def test_mnn_source_rejects_wrong_commit_and_dirty_tree(self):
        # Git subprocesses are a read-only external boundary; no test commits.
        for revision, status in (("0" * 40, ""),
                                 ("3db3cc904dfea55286972b472b040ad5525aa083", " M source/core/Tensor.cpp")):
            with patch.object(qualification, "git_text", side_effect=[str(Path.cwd()), revision, status]):
                with self.assertRaises(ValueError):
                    qualification.verify_mnn_source(Path.cwd())

    def test_cli_persists_numerical_metrics_when_qualification_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fail_after_comparison(args, output, report):
                report["mnn_parity"] = qualification.parity_metrics(
                    np.array([0], np.float32), np.array([0.5], np.float32))
                raise ValueError("MNN/PyTorch parity failed")

            # Only the unavailable conversion/execution boundary is replaced;
            # the numerical comparison and CLI failure-report writer are real.
            with patch.object(qualification, "qualify", side_effect=fail_after_comparison), redirect_stderr(io.StringIO()):
                status = qualification.main([
                    "--stage", "encoder", "--vae-stage-dir", str(root / "source"),
                    "--output-dir", str(root / "out"), "--converter", "/missing/converter",
                    "--runner", "/missing/runner", "--mnn-source", "/missing/source"])
            report = json.loads((root / "out" / "encoder" / "report.json").read_text())
            self.assertEqual(status, 1)
            self.assertFalse(report["mnn_host_parity_passed"])
            self.assertFalse(report["android_gpu_pack_ready"])
            self.assertEqual(report["mnn_parity"]["max_absolute_error"], 0.5)
            self.assertEqual(report["mnn_parity"]["rmse"], 0.5)

    def test_parent_source_dir_accepts_direct_and_stage_nested_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for nested in (False, True):
                base = root / str(nested)
                stage_dir = base / "encoder" if nested else base
                stage_dir.mkdir(parents=True)
                report = source_report(stage_dir)
                report["vae_qualification_passed"] = False
                (stage_dir / "report.json").write_text(json.dumps(report))
                result = subprocess.run([
                    sys.executable, "-B", qualification.__file__, "--stage", "encoder",
                    "--source-dir", str(base), "--output-dir", str(root / "out"),
                    "--converter", "/missing/converter", "--runner", "/missing/runner",
                    "--mnn-source", "/missing/source"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                failed = json.loads((root / "out" / "encoder" / "report.json").read_text())
                self.assertIn("vae_qualification_passed", failed["error"])

    def test_failed_subprocess_preserves_bounded_log_in_negative_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"mnn_host_parity_passed": False, "android_gpu_pack_ready": False}
            args = SimpleNamespace(timeout_seconds=10)
            with self.assertRaises(ValueError):
                qualification.run_command([
                    sys.executable, "-c", "print('x'*20000); print('unsupported operator'); raise SystemExit(7)"],
                    "convert", args, root, report)
            self.assertEqual(report["commands"]["convert"]["returncode"], 7)
            tail = report["commands"]["convert"]["log_tail"]
            self.assertIn("unsupported operator", tail)
            self.assertLessEqual(len(tail), 16384)
            self.assertFalse(report["mnn_host_parity_passed"])

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX unavailable locally; parent CI gate")
    def test_real_onnx_inspection_rejects_half_ports_and_external_data(self):
        import onnx
        from onnx import helper, TensorProto
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            contract = vae.expected_contract("encoder")
            ports = {g: [helper.make_tensor_value_info(n, TensorProto.FLOAT, t["shape"])
                         for n, t in contract[g].items()] for g in ("inputs", "outputs")}
            graph = helper.make_graph([helper.make_node("Identity", ["posterior_epsilon"], ["guide"])],
                                      "port-test", ports["inputs"], ports["outputs"])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            onnx.save(model, path)
            self.assertEqual(qualification.inspect_onnx(path, "encoder"), contract)
            model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
            onnx.save(model, path)
            with self.assertRaises(ValueError):
                qualification.inspect_onnx(path, "encoder")
            model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
            external = model.graph.initializer.add(name="unbound", data_type=TensorProto.FLOAT, dims=[1])
            external.data_location = TensorProto.EXTERNAL
            external.external_data.add(key="location", value="weights.bin")
            path.write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(ValueError, "external"):
                qualification.inspect_onnx(path, "encoder")


if __name__ == "__main__":
    unittest.main()
