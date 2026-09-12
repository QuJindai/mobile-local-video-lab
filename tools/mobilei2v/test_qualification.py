"""Release evidence must reject wrong weights, missing parameters and bad tensors."""
import hashlib
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import qualify_upstream as qualification


class QualificationTest(unittest.TestCase):
    def test_requested_gpu_never_silently_becomes_cpu(self):
        self.assertEqual(qualification.require_execution_device("cpu", False), "cpu")
        self.assertEqual(qualification.require_execution_device("cuda", True), "cuda")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            qualification.require_execution_device("cuda", False)
        with self.assertRaises(ValueError):
            qualification.require_execution_device("unknown", True)

    def test_onnx_cuda_request_requires_cuda_provider(self):
        self.assertEqual(qualification.select_onnx_providers("cpu", ["CPUExecutionProvider"]),
                         ["CPUExecutionProvider"])
        self.assertEqual(qualification.select_onnx_providers("cuda", ["CPUExecutionProvider", "CUDAExecutionProvider"]),
                         ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            qualification.select_onnx_providers("cuda", ["CPUExecutionProvider"])

    def test_checks_bytes_and_hash_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(b"known checkpoint")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            qualification.verify_checkpoint(path, digest, path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                qualification.verify_checkpoint(path, "0" * 64, path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "size"):
                qualification.verify_checkpoint(path, digest, 1)

    def test_all_learned_parameters_must_load(self):
        qualification.validate_loaded_keys(["pos_embed"], [])
        with self.assertRaisesRegex(ValueError, "blocks.0.attn.qkv.weight"):
            qualification.validate_loaded_keys(["blocks.0.attn.qkv.weight"], [])
        with self.assertRaisesRegex(ValueError, "unrecognized.weight"):
            qualification.validate_loaded_keys([], ["unrecognized.weight"])

    def test_parity_rejects_shape_broadcasting_and_nonfinite_values(self):
        actual = np.ones((2, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            qualification.compare_outputs(np.ones((1, 4)), actual)
        for invalid in (np.nan, np.inf, -np.inf):
            bad = actual.copy()
            bad[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "finite"):
                qualification.compare_outputs(actual, bad)

    def test_parity_rejects_large_numerical_error(self):
        expected = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = qualification.compare_outputs(expected, expected + 0.0001)
        self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError, "parity"):
            qualification.compare_outputs(expected, expected + 0.2)

    def test_candidate_cannot_enlarge_reference_relative_tolerance(self):
        reference = np.array([0], dtype=np.float16)
        # This representable FP16 error exceeds 0.005 against reference zero,
        # but was accepted when the candidate itself set the relative budget.
        for value in (0.0050201416015625, -0.0050201416015625):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "parity"):
                qualification.compare_outputs(reference, np.array([value], dtype=np.float16))

    def test_nonfinite_prompt_forward_preserves_failure_report(self):
        # Exercise failure reporting only; no model forward or GPU pass is mocked.
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                report_path = Path(directory) / "report.json"
                def failed_run(args, report):
                    qualification.record_prompt_perturbation(
                        report, np.array([0.0]), np.array([invalid]))
                argv = ["qualify_upstream", "--upstream", directory,
                        "--checkpoint", str(Path(directory) / "weights.pth"),
                        "--onnx", str(Path(directory) / "model.onnx"),
                        "--report", str(report_path)]
                with patch.object(sys, "argv", argv), patch.object(qualification, "run", failed_run), \
                        redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "finite"):
                    qualification.main()
                report = json.loads(report_path.read_text())
                self.assertFalse(report["denoiser_qualification_passed"])
                self.assertFalse(report["android_gpu_pack_ready"])
                self.assertIn("finite", report["error"])
                self.assertNotIn("prompt_perturbation_max_abs", report)

    def test_prompt_invariance_is_measured_and_changed_output_is_rejected(self):
        report = {}
        reference = np.array([1.0, -2.0], dtype=np.float32)
        qualification.record_prompt_perturbation(report, reference, reference.copy())
        self.assertEqual(report["prompt_perturbation_max_abs"], 0.0)
        with self.assertRaisesRegex(ValueError, "prompt affects"):
            qualification.record_prompt_perturbation(report, reference, reference + 0.25)
        self.assertEqual(report["prompt_perturbation_max_abs"], 0.25)

    def test_graph_contract_checks_input_types_shapes_and_output(self):
        valid = qualification.expected_denoiser_contract()
        qualification.validate_graph_contract(valid)
        invalid = {**valid, "latent": {"dtype": "float16", "shape": [1, 128, 3, 23, 40]}}
        with self.assertRaisesRegex(ValueError, "latent"):
            qualification.validate_graph_contract(invalid)
        with self.assertRaisesRegex(ValueError, "output"):
            qualification.validate_graph_contract({k: v for k, v in valid.items() if k != "output"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            qualification.validate_graph_contract({**valid, "prompt": {"dtype": "float16", "shape": [300]}})


if __name__ == "__main__":
    unittest.main()
