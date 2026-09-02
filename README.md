# Mobile Local Video Lab

Android/Snapdragon local-video experiments for fully offline neural video generation.

## V0.3 milestone

V0.3 keeps the validated RIFE v4.6 / ncnn / Vulkan offline generation core and focuses on the remaining handset UX defects found during S24U acceptance.

### Local generation modes

- **Single image → motion clip**: choose Cinematic Auto, Push In, Pan Left, or Drift Up; the app constructs the motion endpoint locally and RIFE generates neural intermediate frames.
- **Two images → interpolation clip**: uses RIFE directly between two user-selected images.

### V0.3 handset workflow

- Replaces the vendor-sensitive embedded `VideoView` result surface with a real MP4 thumbnail extracted by `MediaMetadataRetriever`; tapping the thumbnail hands playback to the system video player.
- Result cards show timestamp, duration, resolution, frame count, and FPS.
- The last five generated results persist across restarts and now render as visual thumbnail rows with metadata.
- V0.2 URI-only history migrates automatically to the V0.3 metadata record format.
- Dynamic single/two-image mode UI, 9/17-frame and 6/8/12-FPS presets, open/share actions, and exportable diagnostics remain available.
- H.264 MP4 encoding writes `YUV_420_888` `Image` planes using each codec plane's row/pixel stride, preserving the vendor-encoder compatibility fix validated on the handset.
- Output is published to `Movies/LocalVideoLab`.

### Build/runtime gates

GitHub Actions builds the arm64 RIFE/ncnn/Vulkan runtime from pinned upstream revisions, packages RIFE v4.6 weights, runs JVM tests, checks 16 KB ELF segment alignment, validates APK version/runtime payload/stable signing identity, and uploads an installable APK.

V0.2 established the **stable development signing baseline** and V0.3 keeps the same certificate, so V0.2+ test builds can overwrite one another. The development signing key is intentionally public in this open-source laboratory repository and must not be reused for production distribution or security-sensitive applications.

All inference and MP4 encoding run on-device with no cloud API. This RIFE milestone is deliberately described as neural interpolation/motion generation rather than diffusion or semantic text-to-video; MobileI2V/QNN remains a later model-backend evolution path.
