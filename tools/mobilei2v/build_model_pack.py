#!/usr/bin/env python3
"""Build and verify Local Video Lab external model packs.

This tool deliberately operates on already-exported runtime artifacts. It does not
pretend that the public PyTorch checkpoint is itself Android-executable.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

FORMAT = "local-video-model-pack-v1"
MAX_FILES = 128
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    pure = PurePosixPath(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts or "," in rel or "\n" in rel:
        raise ValueError(f"unsafe model artifact path: {rel!r}")
    return rel


def list_artifacts(input_dir: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        rel = safe_relative(path, input_dir)
        if rel == "model-pack.properties":
            continue
        result.append((rel, path))
    if not result:
        raise ValueError("input directory contains no model artifacts")
    if len(result) > MAX_FILES:
        raise ValueError(f"too many model artifacts: {len(result)}")
    return result


def render_manifest(args: argparse.Namespace, artifacts: list[tuple[str, Path]]) -> str:
    lines = [
        f"format={FORMAT}",
        f"id={args.id}",
        f"backend={args.backend}",
        f"version={args.version}",
        f"source.repo={args.source_repo}",
        f"source.commit={args.source_commit}",
        f"license.code={args.code_license}",
        f"license.weights={args.weights_license}",
        f"min.ram.mb={args.min_ram_mb}",
        f"recommended.ram.mb={args.recommended_ram_mb}",
        "files=" + ",".join(rel for rel, _ in artifacts),
    ]
    for rel, path in artifacts:
        lines.append(f"sha256.{rel}={sha256_file(path)}")
    return "\n".join(lines) + "\n"


def add_bytes(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data)


def add_file(zf: zipfile.ZipFile, name: str, path: Path) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in {".onnx", ".bin"} else zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with path.open("rb") as handle:
        zf.writestr(info, handle.read())


def build_pack(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    artifacts = list_artifacts(input_dir)
    manifest = render_manifest(args, artifacts)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(temp, "w", allowZip64=True) as zf:
        add_bytes(zf, "model-pack.properties", manifest.encode("utf-8"))
        for rel, path in artifacts:
            add_file(zf, rel, path)
    os.replace(temp, output)
    verify_pack(output)
    return output


def parse_properties(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid property line: {raw!r}")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def verify_pack(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if "model-pack.properties" not in names:
            raise ValueError("model-pack.properties missing")
        props = parse_properties(zf.read("model-pack.properties").decode("utf-8"))
        if props.get("format") != FORMAT:
            raise ValueError("unsupported model-pack format")
        files = [item for item in props.get("files", "").split(",") if item]
        if not files or len(files) > MAX_FILES:
            raise ValueError("invalid artifact list")
        for rel in files:
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts or rel not in names:
                raise ValueError(f"unsafe/missing artifact: {rel}")
            expected = props.get(f"sha256.{rel}", "")
            actual = hashlib.sha256(zf.read(rel)).hexdigest()
            if expected != actual:
                raise ValueError(f"checksum mismatch: {rel}")
        return props


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="mobilei2v-300m")
    parser.add_argument("--backend", default="mobilei2v")
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-repo", default="hustvl/MobileI2V")
    parser.add_argument("--source-commit", default="8d0a253c766b05a43ba408baf5e8f800a36be8b4")
    parser.add_argument("--code-license", default="Apache-2.0")
    parser.add_argument("--weights-license", default="MIT")
    parser.add_argument("--min-ram-mb", type=int, default=8192)
    parser.add_argument("--recommended-ram-mb", type=int, default=12288)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.verify_only:
        props = verify_pack(args.output)
        print(f"PASS {props.get('id')} {props.get('version')} {args.output}")
        return
    output = build_pack(args)
    print(f"BUILT {output}")
    print(f"SHA256 {sha256_file(output)}")


if __name__ == "__main__":
    main()
