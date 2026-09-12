#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import zipfile

from build_gpu_pack import build_pack, parse_manifest, verify_pack


FILES = {
    "denoiser.mnn": b"mobilei2v-denoiser\x00" * 17,
    "vae_encoder.mnn": b"mobilei2v-vae-encoder\x00" * 11,
    "vae_decoder.mnn": b"mobilei2v-vae-decoder\x00" * 13,
    "empty_prompt.f16": b"\x00\x3c" * 96,
    "empty_prompt_mask.bin": b"\x01" * 600,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def populate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in FILES.items():
        (root / name).write_bytes(data)


def test_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        populate(src)
        a = tmp / "a.mlvpkg"
        b = tmp / "b.mlvpkg"
        build_pack(src, a, "0.7.0-test")
        build_pack(src, b, "0.7.0-test")
        assert sha256(a) == sha256(b), "same inputs must create byte-identical pack"
        pa = verify_pack(a)
        pb = verify_pack(b)
        assert pa == pb


def test_manifest_pins_and_runtime_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        populate(src)
        out = tmp / "gpu.mlvpkg"
        build_pack(src, out, "0.7.0-test")
        props = verify_pack(out)
        assert props["format"] == "local-video-model-pack-v2"
        assert props["backend"] == "mobilei2v"
        assert props["execution"] == "mnn-opencl"
        assert props["source.repo"] == "hustvl/MobileI2V"
        assert props["source.commit"] == "8d0a253c766b05a43ba408baf5e8f800a36be8b4"
        assert props["checkpoint.sha256"] == "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d"
        assert props["dream.source"] == "xororz/local-dream"
        assert props["dream.commit"] == "a7666f6198412a58c6eb1eacc28828aa40c7d7ae"
        assert props["mnn.commit"] == "3db3cc904dfea55286972b472b040ad5525aa083"
        assert props["frames"] == "17"
        assert props["width"] == "1280"
        assert props["height"] == "720"
        for name in FILES:
            assert name in props["files"].split(",")
            assert len(props[f"sha256.{name}"]) == 64


def test_missing_runtime_artifact_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        populate(src)
        (src / "vae_decoder.mnn").unlink()
        try:
            build_pack(src, tmp / "bad.mlvpkg", "0.7.0-test")
        except ValueError as exc:
            assert "vae_decoder.mnn" in str(exc)
        else:
            raise AssertionError("pack build accepted a missing VAE decoder")


def test_tampered_artifact_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        populate(src)
        out = tmp / "gpu.mlvpkg"
        build_pack(src, out, "0.7.0-test")
        corrupt = tmp / "corrupt.mlvpkg"
        with zipfile.ZipFile(out, "r") as zf, zipfile.ZipFile(corrupt, "w") as dst:
            for info in zf.infolist():
                data = zf.read(info.filename)
                if info.filename == "denoiser.mnn":
                    data += b"tamper"
                dst.writestr(info, data)
        try:
            verify_pack(corrupt)
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("tampered runtime artifact was accepted")


def test_unsafe_manifest_file_is_rejected() -> None:
    text = "format=local-video-model-pack-v2\nfiles=../evil.mnn\nsha256.../evil.mnn=" + "0" * 64 + "\n"
    try:
        parse_manifest(text)
    except ValueError as exc:
        assert "unsafe" in str(exc) or "path" in str(exc)
    else:
        raise AssertionError("unsafe manifest path was accepted")


def main() -> None:
    tests = [
        test_deterministic,
        test_manifest_pins_and_runtime_contract,
        test_missing_runtime_artifact_fails,
        test_tampered_artifact_fails,
        test_unsafe_manifest_file_is_rejected,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"PASS {len(tests)}/{len(tests)} accelerated GPU pack tests")


if __name__ == "__main__":
    main()
