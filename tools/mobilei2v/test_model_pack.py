#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

import build_model_pack as bmp


class ModelPackToolTest(unittest.TestCase):
    def make_args(self, root: Path, output: Path):
        return argparse.Namespace(
            input_dir=root,
            output=output,
            id="mobilei2v-test",
            backend="mobilei2v",
            version="test-1",
            source_repo="hustvl/MobileI2V",
            source_commit="8d0a253c766b05a43ba408baf5e8f800a36be8b4",
            code_license="Apache-2.0",
            weights_license="MIT",
            min_ram_mb=8192,
            recommended_ram_mb=12288,
            verify_only=False,
        )

    def test_build_is_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "export"
            (src / "models").mkdir(parents=True)
            (src / "constants").mkdir(parents=True)
            (src / "models" / "dit.onnx").write_bytes(b"dit" * 1000)
            (src / "constants" / "cond.bin").write_bytes(b"cond" * 777)
            one = bmp.build_pack(self.make_args(src, base / "one.mlvpkg"))
            two = bmp.build_pack(self.make_args(src, base / "two.mlvpkg"))
            self.assertEqual(bmp.sha256_file(one), bmp.sha256_file(two))
            props = bmp.verify_pack(one)
            self.assertEqual("mobilei2v-test", props["id"])
            self.assertEqual("mobilei2v", props["backend"])

    def test_verifier_detects_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "export"
            src.mkdir()
            (src / "dit.onnx").write_bytes(b"original")
            pack = bmp.build_pack(self.make_args(src, base / "pack.mlvpkg"))
            broken = base / "broken.mlvpkg"
            with zipfile.ZipFile(pack, "r") as zin, zipfile.ZipFile(broken, "w") as zout:
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    if info.filename == "dit.onnx":
                        data = b"tampered"
                    zout.writestr(info, data)
            with self.assertRaises(ValueError):
                bmp.verify_pack(broken)


if __name__ == "__main__":
    unittest.main()
