"""Protect precision experiment identity and distinct reference comparisons."""
import json
import unittest
import numpy as np
import diagnose_precision as diagnosis


class PrecisionDiagnosisTest(unittest.TestCase):
    def test_original_tolerance_records_failure_without_losing_metrics(self):
        reference = np.array([0, 1], dtype=np.float16)
        actual = np.array([0.0050201416015625, 1], dtype=np.float16)
        result = diagnosis.metrics(reference, actual)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_elements"], 1)
        self.assertEqual(result["atol"], 0.005)
        self.assertEqual(result["rtol"], 0.01)
        self.assertEqual(result["worst_index"], [0])
        self.assertGreater(result["worst_tolerance_ratio"], 1)
        json.dumps(result, allow_nan=False)

    def test_wrong_shape_and_nonfinite_data_cannot_be_diagnostic_passes(self):
        for actual in (np.array([[0]]), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                diagnosis.metrics(np.array([0]), actual)

    def test_same_precision_pass_is_distinct_from_original_reference_drift(self):
        original = np.array([0], np.float16)
        promoted = np.array([0.02], np.float32)
        self.assertTrue(diagnosis.metrics(promoted, promoted, atol=1e-4, rtol=1e-3)["passed"])
        self.assertFalse(diagnosis.metrics(original, promoted)["passed"])

    def test_fp32_profile_does_not_mutate_original_contract(self):
        promoted = diagnosis.contract("promoted-fp32")
        self.assertEqual({v["dtype"] for v in promoted.values()}, {"float32"})
        original = diagnosis.contract("original-fp16")
        self.assertEqual(original["latent"]["dtype"], "float16")
        self.assertEqual(original["cond_mask"]["dtype"], "float32")
        with self.assertRaises(ValueError):
            diagnosis.contract("automatic")

    def test_inputs_must_be_exact_expansions_not_fresh_fp32_random_values(self):
        original = {name: np.array([0.1234], np.float16) for name in diagnosis.NAMES}
        promoted = {name: value.astype(np.float32) for name, value in original.items()}
        diagnosis.verify_shared_inputs(original, promoted)
        for bad in (np.array([0.1234], np.float32), np.array([[0.1234]], np.float32),
                    np.array([0.1234], np.float16), np.array([np.nan], np.float32)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                diagnosis.verify_shared_inputs(original, {**promoted, "latent": bad})


if __name__ == "__main__":
    unittest.main()
