"""Release evidence must reject wrong weights, missing parameters and bad tensors."""
import hashlib
import tempfile
import unittest
from pathlib import Path

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
