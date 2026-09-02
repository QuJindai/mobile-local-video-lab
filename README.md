# Mobile Local Video Lab

Android/Snapdragon experiments for fully offline neural video generation.

## V0.4 milestone

V0.3 is the S24U-validated RIFE baseline. V0.4 turns that single-backend app into a **pluggable local-video model workbench** so semantic I2V models can be installed and evolved without rewriting the handset UX/export pipeline.

### Backends

- **RIFE Motion — READY / handset validated**: single-image camera-motion clips and two-image neural interpolation using RIFE v4.6 / ncnn / Vulkan. This remains the built-in safe fallback and continues to export H.264 MP4 to `Movies/LocalVideoLab`.
- **MobileI2V 0.27B — deployment track**: a separate semantic image-to-video backend based on `hustvl/MobileI2V`. V0.4 introduces its external model-pack, compatibility and ONNX Runtime foundations. The app does **not** call RIFE when MobileI2V is selected and does not claim semantic generation until a genuine MobileI2V execution runtime passes all readiness gates.

### Model Lab

The V0.4 Android UI exposes backend selection, handset capability information and MobileI2V pack management. MobileI2V artifacts are installed as a versioned `.mlvpkg` in app-private storage instead of inflating the APK by roughly a gigabyte or more.

The `local-video-model-pack-v1` installer:

- copies the selected pack through a streaming path rather than loading it into the Java heap;
- rejects absolute paths and ZIP/path traversal;
- validates a versioned manifest and per-artifact SHA-256 values;
- checks available private storage before expansion;
- extracts only declared artifacts;
- activates a verified version atomically and restores the prior version on activation failure;
- stores source commit and code/weights license metadata alongside the installed artifacts.

### MobileI2V deployment baseline

V0.4 pins:

- upstream source `hustvl/MobileI2V` commit `8d0a253c766b05a43ba408baf5e8f800a36be8b4`;
- architecture `Mobiledit_300M_P1_D16`;
- public `hybrid_371.pth` checkpoint identity and SHA-256;
- native research target `1280×720×17`;
- ONNX Runtime Android `1.29.0` as the first generic Android execution foundation.

The official public repository provides PyTorch/CUDA inference and a mobile demonstration but not a reusable Android runtime implementation. Therefore V0.4 keeps `MobileI2V runtime pending` as a visible state until exported graphs and the actual Android execution loop are proven. See `docs/V0.4_MODEL_BACKEND_PLAN.md` and `tools/mobilei2v/`.

### RIFE handset workflow retained

- Cinematic Auto, Push In, Pan Left and Drift Up single-image motion presets.
- Two-image interpolation.
- 9/17 frame and 6/8/12 FPS presets.
- Vendor-safe H.264 input through `YUV_420_888` plane row/pixel strides.
- Real MP4 thumbnails, system full-screen playback and sharing.
- Last five results persist with thumbnails and metadata.
- Exportable diagnostics, thermal and memory telemetry.

### Build/runtime gates

GitHub Actions now gates both the stable and evolving layers: deterministic model-pack build/tamper tests, JVM contracts for model-pack parsing/path safety/backend routing, the existing RIFE/ncnn/Vulkan build, 16 KB ELF compatibility, V0.4 APK version and stable signing identity, and packaged ONNX Runtime arm64 libraries.

The V0.2+ stable development certificate is retained, so later laboratory APKs can overwrite the validated V0.3 install. The development key is public by design for this open-source laboratory repository and must not be used as a production signing identity.

See `THIRD_PARTY_NOTICES.md` for source/model/runtime attribution boundaries.
