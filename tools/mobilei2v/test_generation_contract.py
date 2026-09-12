#!/usr/bin/env python3
"""Run the actual APK source gate against controlled source mutations.

These are CI-gate regression tests, not Android or GPU inference tests.
"""
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("app/src/main/java/com/qujindai/localvideo/MainActivityV05.java")
ENGINE = Path("app/src/main/java/com/qujindai/localvideo/MobileI2VGpuEngine.java")
SOURCES = (
    MAIN,
    ENGINE,
    Path("app/src/main/java/com/qujindai/localvideo/MobileI2VMicroscope.java"),
    Path("app/src/main/cpp/mobilei2v/mobilei2v_jni.cpp"),
)


def workflow_gate():
    workflow = (ROOT / ".github/workflows/android-apk.yml").read_text(encoding="utf-8")
    name = "      - name: Verify MobileI2V generation wiring\n"
    step = workflow.split(name, 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class GenerationContractTest(unittest.TestCase):
    def run_gate(self, replacements=None):
        with tempfile.TemporaryDirectory(prefix="mobilei2v-contract-") as temp:
            root = Path(temp)
            for path in SOURCES:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                content = (ROOT / path).read_text(encoding="utf-8")
                if replacements and path in replacements:
                    content = replacements[path](content)
                destination.write_text(content, encoding="utf-8")
            return subprocess.run(
                ["bash", "-c", workflow_gate()], cwd=root,
                capture_output=True, text=True, timeout=15,
            )

    def test_fixed_frame_control_does_not_block_generation_gate(self):
        result = self.run_gate()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_force_disabled_generate_button_is_rejected(self):
        def disable_button(source):
            return re.sub(
                r"generateButton\.setEnabled\([^;]+;",
                "generateButton.setEnabled(primaryUri != null && decision.ready && !busy && !mobile);",
                source,
            ).replace("framesSpinner.setEnabled(!busy && !mobile)",
                      "framesSpinner.setEnabled(!busy)")

        result = self.run_gate({MAIN: disable_button})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("force-disabled", result.stderr)

    def test_multiline_force_disabled_button_is_rejected(self):
        def disable_button(source):
            return re.sub(
                r"generateButton\.setEnabled\([^;]+;",
                "generateButton.setEnabled(\n    primaryUri != null && decision.ready && !busy\n    && ! mobile);",
                source,
            ).replace("framesSpinner.setEnabled(!busy && !mobile)",
                      "framesSpinner.setEnabled(!busy)")

        result = self.run_gate({MAIN: disable_button})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("force-disabled", result.stderr)

    def test_missing_gpu_decode_call_is_rejected(self):
        result = self.run_gate({ENGINE: lambda s: s.replace("session.decode(sampled)", "skipDecode(sampled)")})
        self.assertNotEqual(0, result.returncode)

    def test_rife_substitution_is_rejected(self):
        result = self.run_gate({ENGINE: lambda s: s + "\nRifeEngine fallback;\n"})
        self.assertNotEqual(0, result.returncode)
        self.assertIn("RIFE must not appear", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
