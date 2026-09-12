#!/usr/bin/env python3
"""Build and verify deterministic MobileI2V accelerated Android model packs.

This module never converts PyTorch weights. It only packages *already exported
and parity-verified* MNN runtime artifacts. `export_mobilei2v_gpu.py` is
responsible for producing those artifacts from the pinned upstream checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import os
import tempfile
import zipfile

FORMAT = "local-video-model-pack-v2"
PACK_ID = "mobilei2v-300m-gpu"
BACKEND = "mobilei2v"
EXECUTION = "mnn-opencl"
SOURCE_REPO = "hustvl/MobileI2V"
SOURCE_COMMIT = "8d0a253c766b05a43ba408baf5e8f800a36be8b4"
CHECKPOINT_SHA256 = "bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d"
DREAM_SOURCE = "xororz/local-dream"
DREAM_COMMIT = "a7666f6198412a58c6eb1eacc28828aa40c7d7ae"
MNN_COMMIT = "3db3cc904dfea55286972b472b040ad5525aa083"
FRAMES = 17
WIDTH = 1280
HEIGHT = 720
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
CHUNK = 1024 * 1024
MAX_FILES = 128

REQUIRED = (
    "denoiser.mnn",
    "vae_encoder.mnn",
    "vae_decoder.mnn",
    "empty_prompt.f16",
    "empty_prompt_mask.bin",
)
RUNTIME_PROPERTIES = "runtime.properties"
MANIFEST = "model-pack.properties"


def _safe_path(path: str) -> str:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in path
        or "," in path
        or "\n" in path
        or "\r" in path
    ):
        raise ValueError(f"unsafe model artifact path: {path!r}")
    return pure.as_posix()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(text: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid property line: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError("empty property key")
        props[key] = value

    file_list = props.get("files")
    if file_list is not None:
        files = [part.strip() for part in file_list.split(",") if part.strip()]
        if not files or len(files) > MAX_FILES:
            raise ValueError("invalid accelerated artifact list")
        if len(files) != len(set(files)):
            raise ValueError("duplicate accelerated artifact path")
        for path in files:
            _safe_path(path)
    return props


def _runtime_properties(version: str, vae_impl: str) -> bytes:
    text = (
        f"runtime.format=mobilei2v-android-runtime-v1\n"
        f"runtime.execution={EXECUTION}\n"
        f"runtime.version={version}\n"
        f"model.frames={FRAMES}\n"
        f"model.width={WIDTH}\n"
        f"model.height={HEIGHT}\n"
        "model.latent.channels=128\n"
        "model.latent.frames=3\n"
        "model.latent.height=23\n"
        "model.latent.width=40\n"
        "model.cfg.batch=2\n"
        "model.prompt.length=300\n"
        "model.prompt.channels=896\n"
        "sampler=flow-euler\n"
        "sampler.flow.shift=3.0\n"
        f"vae.impl={vae_impl}\n"
        f"dream.commit={DREAM_COMMIT}\n"
        f"mnn.commit={MNN_COMMIT}\n"
    )
    return text.encode("utf-8")


def _collect(input_dir: Path, version: str, vae_impl: str) -> list[tuple[str, Path | None, bytes | None]]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"export directory does not exist: {input_dir}")
    for required in REQUIRED:
        path = input_dir / required
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"required accelerated artifact missing: {required}")

    items: list[tuple[str, Path | None, bytes | None]] = []
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        rel = _safe_path(path.relative_to(input_dir).as_posix())
        if rel in {MANIFEST, RUNTIME_PROPERTIES}:
            continue
        items.append((rel, path, None))
    items.append((RUNTIME_PROPERTIES, None, _runtime_properties(version, vae_impl)))
    if len(items) > MAX_FILES:
        raise ValueError(f"too many accelerated artifacts: {len(items)}")
    return sorted(items, key=lambda item: item[0])


def _render_manifest(
    items: list[tuple[str, Path | None, bytes | None]], version: str
) -> bytes:
    names = [name for name, _, _ in items]
    hashes: dict[str, str] = {}
    for name, path, data in items:
        hashes[name] = _sha_file(path) if path is not None else _sha_bytes(data or b"")
    lines = [
        f"format={FORMAT}",
        f"id={PACK_ID}",
        f"backend={BACKEND}",
        f"version={version}",
        f"execution={EXECUTION}",
        f"source.repo={SOURCE_REPO}",
        f"source.commit={SOURCE_COMMIT}",
        f"checkpoint.sha256={CHECKPOINT_SHA256}",
        f"dream.source={DREAM_SOURCE}",
        f"dream.commit={DREAM_COMMIT}",
        f"mnn.commit={MNN_COMMIT}",
        f"frames={FRAMES}",
        f"width={WIDTH}",
        f"height={HEIGHT}",
        "files=" + ",".join(names),
    ]
    lines.extend(f"sha256.{name}={hashes[name]}" for name in names)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _zip_info(name: str, stored: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _write_bytes(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    zf.writestr(_zip_info(name, stored=False), data)


def _write_file(zf: zipfile.ZipFile, name: str, path: Path) -> None:
    stored = path.suffix.lower() in {".mnn", ".bin", ".f16", ".weight"}
    with path.open("rb") as source, zf.open(_zip_info(name, stored=stored), "w", force_zip64=True) as dest:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            dest.write(chunk)


def build_pack(input_dir: Path, output: Path, version: str, vae_impl: str = "upstream") -> Path:
    if not version or any(c not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz._-" for c in version):
        raise ValueError(f"unsafe pack version: {version!r}")
    if vae_impl not in {"upstream", "turbo-vaed"}:
        raise ValueError(f"unsupported VAE identity: {vae_impl}")
    items = _collect(input_dir, version, vae_impl)
    manifest = _render_manifest(items, version)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with zipfile.ZipFile(temp, "w", allowZip64=True) as zf:
            _write_bytes(zf, MANIFEST, manifest)
            for name, path, data in items:
                if path is not None:
                    _write_file(zf, name, path)
                else:
                    _write_bytes(zf, name, data or b"")
        verify_pack(temp)
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()
    return output


def _require(props: dict[str, str], key: str, expected: str) -> None:
    actual = props.get(key)
    if actual != expected:
        raise ValueError(f"unexpected {key}: {actual!r}; expected {expected!r}")


def verify_pack(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP entries in accelerated model pack")
        if MANIFEST not in names:
            raise ValueError("model-pack.properties missing")
        props = parse_manifest(zf.read(MANIFEST).decode("utf-8"))
        _require(props, "format", FORMAT)
        _require(props, "backend", BACKEND)
        _require(props, "execution", EXECUTION)
        _require(props, "source.repo", SOURCE_REPO)
        _require(props, "source.commit", SOURCE_COMMIT)
        _require(props, "checkpoint.sha256", CHECKPOINT_SHA256)
        _require(props, "dream.source", DREAM_SOURCE)
        _require(props, "dream.commit", DREAM_COMMIT)
        _require(props, "mnn.commit", MNN_COMMIT)
        _require(props, "frames", str(FRAMES))
        _require(props, "width", str(WIDTH))
        _require(props, "height", str(HEIGHT))

        files = [part.strip() for part in props.get("files", "").split(",") if part.strip()]
        for required in (*REQUIRED, RUNTIME_PROPERTIES):
            if required not in files:
                raise ValueError(f"required accelerated artifact missing: {required}")
        for name in files:
            _safe_path(name)
            if name not in names:
                raise ValueError(f"manifest artifact missing from ZIP: {name}")
            expected = props.get(f"sha256.{name}", "")
            if len(expected) != 64:
                raise ValueError(f"invalid checksum declaration: {name}")
            with zf.open(name, "r") as handle:
                actual = _sha_stream(handle)
            if actual != expected:
                raise ValueError(f"checksum mismatch: {name}")
        extra = set(names) - {MANIFEST, *files}
        if extra:
            raise ValueError(f"undeclared ZIP entries: {sorted(extra)}")
        return props


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--vae-impl", choices=["upstream", "turbo-vaed"], default="upstream")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        props = verify_pack(args.output)
        print("PASS", props["id"], props["version"], args.output)
        return
    out = build_pack(args.input_dir, args.output, args.version, args.vae_impl)
    print("BUILT", out)
    print("SHA256", _sha_file(out))


if __name__ == "__main__":
    main()
