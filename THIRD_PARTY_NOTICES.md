# Third-party notices

Local Video Lab combines or interoperates with third-party components. This file documents the V0.4 development baseline and keeps code/model licensing boundaries explicit.

## RIFE / ncnn Vulkan baseline

The built-in handset-validated interpolation runtime is based on the open-source `rife-ncnn-vulkan` / ncnn ecosystem. Its source/runtime provenance remains pinned by the existing runtime preparation script and manifest. Preserve upstream license and attribution terms when redistributing derived binaries.

## MobileI2V

- Source: `hustvl/MobileI2V`
- V0.4 pinned source commit: `8d0a253c766b05a43ba408baf5e8f800a36be8b4`
- Repository source license: Apache License 2.0 (`LICENSE.txt` in the upstream repository)
- Public checkpoint baseline: `hybrid_371.pth`
- Checkpoint SHA-256: `bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d`
- Hugging Face model-card license field at the V0.4 baseline: MIT

MobileI2V weights are **not** embedded in the APK. Exported/compiled artifacts are installed as a separate `.mlvpkg`, whose manifest records the source, source commit, code license, weights license, and per-file SHA-256 values.

## ONNX Runtime Android

V0.4 packages `com.microsoft.onnxruntime:onnxruntime-android:1.29.0` as the first generic Android execution foundation for future semantic I2V graphs. ONNX Runtime is distributed under the MIT License. V0.4 only treats its JNI presence as a runtime foundation; it does not claim that the MobileI2V graph itself is implemented until that execution path is completed and gated.

## Model-To-NPU research reference

`VitalikDen0/Model-To-NPU` was reviewed as an architectural reference for Snapdragon/QNN deployment patterns. Its repository identifies PolyForm Noncommercial License 1.0.0. No source from that repository is copied into Local Video Lab V0.4; this project keeps its implementation independent.
