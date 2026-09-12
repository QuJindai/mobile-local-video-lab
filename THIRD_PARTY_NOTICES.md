# Third-party notices

Local Video Lab combines or interoperates with third-party components. This file documents the V0.5 development baseline and keeps code/model licensing boundaries explicit.

## RIFE / ncnn Vulkan baseline

The built-in handset-validated interpolation runtime is based on the open-source `rife-ncnn-vulkan` / ncnn ecosystem. Its source/runtime provenance remains pinned by the existing runtime preparation script and manifest. Preserve upstream license and attribution terms when redistributing derived binaries.

## Depth Anything V2 Small ONNX

The V0.5 Depth 3D backend packages the Q4 ONNX export from `onnx-community/depth-anything-v2-small`.

- Model repository: `onnx-community/depth-anything-v2-small`
- Repository/model-card license: Apache-2.0
- Packaged artifact: `onnx/model_q4.onnx`
- Packaged artifact SHA-256: `5d55b02762e1907589158af3e366bd61ddf648155852a07bbf5e3a074639fcf8`
- Input used by Local Video Lab: RGB ImageNet-normalized float tensor, `1×3×518×518`
- Output used by Local Video Lab: monocular depth map, normalized locally with robust percentile clipping before parallax synthesis

The model runs through ONNX Runtime Android. Its depth result is used to construct a depth-dependent camera-motion endpoint; RIFE then generates the intermediate video frames. This is a real two-model local pipeline, but it is not described as semantic diffusion I2V.

An INT8 export from the same repository was evaluated during V0.5 development and rejected because its `ConvInteger` graph was not executable on the standard ONNX Runtime CPU path used by the app. It is not packaged in the final V0.5 APK.

## MobileI2V

- Source: `hustvl/MobileI2V`
- V0.5 pinned source commit: `8d0a253c766b05a43ba408baf5e8f800a36be8b4`
- Repository source license: Apache License 2.0 (`LICENSE.txt` in the upstream repository)
- Public checkpoint baseline: `hybrid_371.pth`
- Checkpoint SHA-256: `bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d`
- Hugging Face model-card license field at the V0.5 baseline: MIT

MobileI2V weights are **not** embedded in the APK. Exported/compiled artifacts are installed as a separate `.mlvpkg`, whose manifest records source identity, code/weights license metadata, and per-file SHA-256 values. The app does not report MobileI2V as ready until a genuine Android execution loop is implemented and passes its readiness gates.

## ONNX Runtime Android

V0.5 packages `com.microsoft.onnxruntime:onnxruntime-android:1.29.0`. It is actively used by the Depth Anything V2 backend and also serves as the generic execution foundation for future semantic I2V graph work. ONNX Runtime is distributed under the MIT License.

## Model-To-NPU research reference

`VitalikDen0/Model-To-NPU` was reviewed as an architectural reference for Snapdragon/QNN deployment patterns. Its repository identifies PolyForm Noncommercial License 1.0.0. No source from that repository is copied into Local Video Lab; this project keeps its implementation independent.
