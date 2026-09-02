# Mobile Local Video Lab

Android/Snapdragon experiments for fully offline neural video generation.

## V0.5 milestone

V0.3 is the S24U-validated RIFE baseline. V0.5 expands it into a **three-backend local-video workbench** and adds a second genuinely executable local model path before exposing the still-in-development semantic MobileI2V backend.

### Backends

- **RIFE Motion — READY / S24U validated**: single-image 2D camera motion and two-image neural interpolation through RIFE v4.6 / ncnn / Vulkan.
- **Depth 3D Motion — READY in build gates**: Depth Anything V2 Small Q4 runs through ONNX Runtime at 518×518 to estimate monocular depth. Local Video Lab converts that depth map into depth-dependent parallax/dolly geometry, then RIFE generates continuous frames. Presets: 3D left parallax, 3D right parallax, 3D dolly-in.
- **MobileI2V 0.27B — deployment track / NOT READY**: semantic image-to-video backend based on `hustvl/MobileI2V`. Model-pack plumbing, device capability gates and ONNX Runtime foundations are present; the app will not claim this backend is ready until its real Android graph execution loop passes the same gates.

Depth 3D is intentionally described as a two-model depth-aware motion pipeline rather than diffusion I2V. Selecting MobileI2V never silently falls back to RIFE.

### Depth 3D pipeline

```text
input photo
  -> Depth Anything V2 Small Q4 / ONNX Runtime
  -> robust normalized depth map
  -> depth-dependent parallax/dolly endpoint
  -> RIFE v4.6 / ncnn / Vulkan neural intermediate frames
  -> vendor-safe H.264 MediaCodec encode
  -> Movies/LocalVideoLab
```

The Q4 model is downloaded from the pinned `onnx-community/depth-anything-v2-small` artifact during CI, SHA-256 verified, host-inference-smoked with ONNX Runtime, then packaged in the APK. The INT8 export was explicitly rejected during development because its `ConvInteger` path failed on the standard ONNX Runtime CPU execution provider.

### Model Lab / MobileI2V

The handset UI exposes backend readiness, SoC/ABI/RAM/page-size/Vulkan/storage information and MobileI2V pack management. MobileI2V artifacts are installed as a versioned `.mlvpkg` in app-private storage instead of inflating the APK by roughly a gigabyte or more.

The `local-video-model-pack-v1` installer streams large packs, rejects traversal/absolute paths, validates a versioned manifest and per-artifact SHA-256 values, checks free storage, extracts only declared artifacts and atomically activates a verified version with rollback.

V0.5 pins MobileI2V source commit `8d0a253c766b05a43ba408baf5e8f800a36be8b4`, architecture `Mobiledit_300M_P1_D16`, public `hybrid_371.pth` identity, and ONNX Runtime Android 1.29.0 as the generic execution foundation. See `docs/V0.4_MODEL_BACKEND_PLAN.md` and `tools/mobilei2v/`.

### Retained handset workflow

- 9/17 frame and 6/8/12 FPS presets.
- RIFE Cinematic Auto, Push In, Pan Left and Drift Up presets.
- Depth 3D left/right parallax and dolly-in presets.
- `YUV_420_888` row/pixel-stride-safe H.264 encoding.
- Real MP4 thumbnails, system full-screen playback and sharing.
- Last five results persist with thumbnails and metadata.
- Exportable diagnostics, thermal and memory telemetry.

### Build/runtime gates

GitHub Actions verifies deterministic MobileI2V pack tooling, Depth Anything Q4 artifact SHA-256 and host inference, JVM backend/depth/model-pack tests, RIFE/ncnn/Vulkan native build, all packaged arm64 ELF 16 KB compatibility, V0.5 APK version/launcher, stable signing identity, RIFE weights, Depth Anything model and ONNX Runtime JNI libraries.

The V0.2+ stable development certificate is retained, so V0.5 can overwrite V0.3. The development key is public by design for this open-source laboratory repository and must not be used as a production signing identity.

See `THIRD_PARTY_NOTICES.md` for source/model/runtime attribution boundaries.
