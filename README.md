# Mobile Local Video Lab

Android/Snapdragon local-video experiments for fully offline neural video generation.

## V0.2 milestone

V0.2 packages the open-source RIFE v4.6 / ncnn / Vulkan inference core into an installable Android APK and focuses on a complete handset workflow rather than an engineering demo.

### Local generation modes

- **Single image → motion clip**: synthesizes a restrained camera-motion endpoint locally, then uses RIFE to generate neural intermediate frames.
- **Two images → interpolation clip**: uses RIFE to generate intermediate frames between two user-selected images.

### V0.2 product workflow

- Dynamic single/two-image mode UI with clearer primary actions.
- 9/17 frame and 6/8/12 FPS presets.
- In-app video preview after generation.
- Open and Android share actions backed by the saved MediaStore URI.
- Last five generated results persist across app restarts.
- Runtime details are collapsed by default; failure diagnostics can be exported through the Android share sheet.
- H.264 MP4 encoding writes `YUV_420_888` `Image` planes using each codec plane's row/pixel stride instead of assuming planar I420 layout, improving vendor encoder compatibility.
- Output is published to `Movies/LocalVideoLab`.

### Build/runtime gates

GitHub Actions builds the arm64 RIFE/ncnn/Vulkan runtime from pinned upstream revisions, packages RIFE v4.6 weights, runs JVM tests, checks 16 KB ELF segment alignment, validates APK version/runtime payload, and uploads an installable debug APK.

All inference and MP4 encoding run on-device with no cloud API. This RIFE milestone is deliberately described as neural interpolation/motion generation rather than diffusion or semantic text-to-video; MobileI2V/QNN remains a later model-backend evolution path.
