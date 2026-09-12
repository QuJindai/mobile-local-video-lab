"""Torch-free checks for the baseline VAE's sampling, crop, and evidence gates."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import qualify_vae as qualification
except ModuleNotFoundError as error:
    if error.name != "qualify_vae":
        raise
    qualification = None


class VAEQualificationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(qualification, "qualify_vae implementation is missing")

    def test_checkpoint_rejects_wrong_size_and_same_size_corruption(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(b"known weights")
            digest = hashlib.sha256(b"known weights").hexdigest()
            identity = qualification.verify_checkpoint(path, digest, 13)
            self.assertEqual(identity["sha256"], digest)
            self.assertEqual(identity["bytes"], 13)
            with self.assertRaisesRegex(ValueError, "size"):
                qualification.verify_checkpoint(path, digest, 14)
            path.write_bytes(b"wrong weights")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                qualification.verify_checkpoint(path, digest, 13)

    def test_auxiliary_lock_rejects_other_sampling_scale_and_weights(self):
        source = Path(__file__).with_name("auxiliary.lock.properties").read_text()
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            path = Path(directory) / "lock.properties"
            path.write_text(source)
            qualification.read_lock(path)
            for old, new in (("vae.posterior=sample", "vae.posterior=mean"),
                             ("vae.scaling_factor=1.0", "vae.scaling_factor=0.41407"),
                             ("vae.weights.bytes=1676798532", "vae.weights.bytes=13")):
                with self.subTest(new=new):
                    path.write_text(source.replace(old, new))
                    with self.assertRaisesRegex(ValueError, "lock"):
                        qualification.read_lock(path)

    def test_config_rejects_incompatible_shape_and_new_decoder_features(self):
        config = {
            "_class_name": "AutoencoderKLLTXVideo", "_diffusers_version": "0.32.0.dev0",
            "block_out_channels": [128, 256, 512, 512], "decoder_causal": False,
            "encoder_causal": True, "in_channels": 3, "latent_channels": 128,
            "layers_per_block": [4, 3, 3, 3, 4], "out_channels": 3,
            "patch_size": 4, "patch_size_t": 1, "resnet_norm_eps": 1e-6,
            "scaling_factor": 1.0, "spatio_temporal_scaling": [True, True, True, False],
        }
        qualification.validate_config(config)
        for key, value in (("latent_channels", 16), ("patch_size", 2),
                           ("scaling_factor", 0.41407), ("decoder_causal", True),
                           ("timestep_conditioning", True)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "config"):
                    qualification.validate_config({**config, key: value})

    def test_weight_coverage_has_no_missing_buffer_or_extra_key_allowlist(self):
        qualification.validate_loaded_keys([], [])
        for missing, unexpected in ((["encoder.conv_in.conv.weight"], []),
                                    (["latents_mean"], []),
                                    ([], ["decoder.unrecognized.weight"])):
            with self.assertRaisesRegex(ValueError, "coverage"):
                qualification.validate_loaded_keys(missing, unexpected)

    def test_posterior_uses_explicit_epsilon_without_scale_or_normalization(self):
        mean = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        logvar = np.zeros(3, dtype=np.float32)
        epsilon = np.array([2.0, -3.0, 0.25], dtype=np.float32)
        sample = qualification.sample_posterior_numpy(mean, logvar, epsilon)
        np.testing.assert_array_equal(sample, np.array([3.0, -5.0, 3.25], dtype=np.float32))
        np.testing.assert_array_equal(sample, qualification.sample_posterior_numpy(mean, logvar, epsilon))
        np.testing.assert_array_equal(
            qualification.sample_posterior_numpy(mean, logvar, -epsilon),
            np.array([-1.0, 1.0, 2.75], dtype=np.float32))

    def test_logvariance_clamps_before_half_exponent(self):
        mean = np.zeros(4, dtype=np.float32)
        logvar = np.array([-100.0, -30.0, 20.0, 100.0], dtype=np.float32)
        sample = qualification.sample_posterior_numpy(mean, logvar, np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(sample, [3.0590232e-7, 3.0590232e-7, 22026.4658, 22026.4658],
                                   rtol=1e-6, atol=0)
        self.assertEqual(sample.dtype, np.float32)

    def test_posterior_rejects_broadcasting_nonfinite_and_non_fp32_inputs(self):
        values = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            qualification.sample_posterior_numpy(values, values, values[:1])
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                qualification.sample_posterior_numpy(values, values, np.full_like(values, bad))
        with self.assertRaisesRegex(ValueError, "float32"):
            qualification.sample_posterior_numpy(values, values, values.astype(np.float16))

    def test_crop_keeps_top_720_rows_without_resizing_time_or_width(self):
        rows = np.arange(736, dtype=np.float32).reshape(1, 1, 1, 736, 1)
        raw = np.broadcast_to(rows, (1, 3, 17, 736, 1280))
        cropped = qualification.crop_video(raw)
        self.assertEqual(cropped.shape, (1, 3, 17, 720, 1280))
        np.testing.assert_array_equal(cropped[0, 2, 16, :, 1279], np.arange(720, dtype=np.float32))
        self.assertTrue(np.shares_memory(raw, cropped))
        with self.assertRaisesRegex(ValueError, "shape"):
            qualification.crop_video(cropped)

    def test_parity_rejects_broadcasts_nonfinite_and_bad_tolerances(self):
        reference = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            qualification.compare_outputs(reference, reference[:1])
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                qualification.compare_outputs(reference, np.full_like(reference, bad))
        for atol, rtol in ((-1, 0), (0, np.inf), (np.nan, 0)):
            with self.assertRaisesRegex(ValueError, "tolerance"):
                qualification.compare_outputs(reference, reference, atol=atol, rtol=rtol)

    def test_parity_records_numerical_failure_and_chunked_error_metrics(self):
        reference = np.zeros(1048577, dtype=np.float32)
        actual = reference.copy()
        actual[-1] = 0.25
        result = qualification.compare_outputs(reference, actual, atol=0.001, rtol=0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_absolute_error"], 0.25)
        self.assertAlmostEqual(result["rmse"], 0.000244140508584, places=12)
        self.assertTrue(qualification.compare_outputs(reference, reference)["passed"])

    def test_graph_requires_float32_explicit_epsilon_and_exact_crop_shape(self):
        for stage in ("encoder", "decoder"):
            valid = qualification.expected_contract(stage)
            qualification.validate_graph_contract(stage, valid)
            bad = copy.deepcopy(valid)
            output = "guide" if stage == "encoder" else "video"
            bad["outputs"][output]["shape"][-2] -= 1
            with self.assertRaisesRegex(ValueError, "contract"):
                qualification.validate_graph_contract(stage, bad)
        encoder = qualification.expected_contract("encoder")
        del encoder["inputs"]["posterior_epsilon"]
        with self.assertRaisesRegex(ValueError, "contract"):
            qualification.validate_graph_contract("encoder", encoder)
        decoder = qualification.expected_contract("decoder")
        decoder["inputs"]["latent"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "contract"):
            qualification.validate_graph_contract("decoder", decoder)

    def test_inputs_preserve_supplied_epsilon_and_reject_nonfinite_latent(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            path = Path(directory) / "inputs.npz"
            epsilon = np.full((1, 128, 1, 23, 40), 2.5, dtype=np.float32)
            image = np.zeros((1, 3, 720, 1280), dtype=np.float32)
            np.savez(path, image=image, posterior_epsilon=epsilon)
            result = qualification.make_inputs(SimpleNamespace(stage="encoder", inputs=path, seed=999))
            np.testing.assert_array_equal(result["posterior_epsilon"], epsilon)
            image[0, 0, 0, 0] = 1.01
            np.savez(path, image=image, posterior_epsilon=epsilon)
            with self.assertRaisesRegex(ValueError, "normalized"):
                qualification.make_inputs(SimpleNamespace(stage="encoder", inputs=path, seed=1))
            latent = np.zeros((1, 128, 3, 23, 40), dtype=np.float32)
            latent[0, 0, 0, 0, 0] = np.nan
            np.savez(path, latent=latent)
            with self.assertRaisesRegex(ValueError, "finite"):
                qualification.make_inputs(SimpleNamespace(stage="decoder", inputs=path, seed=1))

    def test_ort_rejects_changed_artifact_before_importing_runtime(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            path = root / "model.onnx"
            path.write_bytes(b"exported graph")
            report = {"export_passed": True, "artifacts": {"model.onnx": qualification.file_identity(path)}}
            path.write_bytes(b"modified graph")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                qualification.ort_stage(SimpleNamespace(stage="encoder"), root, report)

    def test_ort_rejects_export_from_different_script_and_keeps_negative_report(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            stage_dir = root / "decoder"
            stage_dir.mkdir()
            report_path = stage_dir / "report.json"
            report_path.write_text(json.dumps({
                "stage": "decoder", "expected_contract": qualification.expected_contract("decoder"),
                "export_passed": True, "vae_qualification_passed": True,
                "android_gpu_pack_ready": False,
                "auxiliary_lock": qualification.file_identity(qualification.LOCK_PATH),
                "script": {"sha256": "0" * 64, "bytes": 0},
            }))
            result = subprocess.run([sys.executable, "-B", qualification.__file__, "--stage", "decoder",
                                     "--vae-dir", str(root), "--output-dir", str(root), "--phase", "ort"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(report_path.read_text())
            self.assertIn("script changed", report["error"])
            self.assertFalse(report["vae_qualification_passed"])
            self.assertFalse(report["android_gpu_pack_ready"])

    def test_cli_rejects_unverified_bytes_before_importing_torch_and_reports_failure(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            root = Path(directory)
            vae = root / "vae"
            vae.mkdir()
            (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"wrong weights")
            command = [sys.executable, "-B", str(Path(qualification.__file__)), "--stage", "encoder",
                       "--vae-dir", str(vae), "--output-dir", str(root / "output"), "--phase", "export"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((root / "output" / "encoder" / "report.json").read_text())
            self.assertIn("size", report["error"])
            self.assertFalse(report["vae_qualification_passed"])
            self.assertFalse(report["android_gpu_pack_ready"])
            self.assertNotIn("No module named", report["error"])


if __name__ == "__main__":
    unittest.main()
