# Face-RE V0.9 Static-Only APK Design

## Goal
Build a single-purpose Android APK for Samsung S24U that performs fully local static image face replacement only.

## Scope
The APK contains one workflow: import user-supplied face models, choose a source-face image, choose a target image, run local inference, preview the result, save/share the result.

The APK must not expose or package RIFE, Depth 3D, MobileI2V, video generation, video decoding, audio remux, motion controls, or their native/model assets.

## Packaging
A new `:facere` application module becomes the only included Gradle app module on `feat/v090-face-re`. The historical `app/` source tree remains in Git history/repository for reference but is not part of the V0.9 APK build.

Application ID remains `com.qujindai.localvideo` so the lab signing chain can be retained. Version is `0.9.0`, versionCode `11`. The existing stable development signing certificate must remain unchanged.

## Runtime
Use ONNX Runtime Android 1.29.0 on arm64-v8a. Initial execution uses optimized CPU sessions with up to four intra-op threads for deterministic compatibility on S24U.

The model contract is compatible with the proven Android pipeline from `Parasaran-Python/android-face-fusion` (MIT): SCRFD detector, ArcFace recognizer, INSwapper-128 swapper, and extracted EMAP. Production model weights are not committed to this public repository or bundled in the APK.

## Model import
The user imports one ZIP model pack containing exactly these required artifacts:

- `det_10g.onnx`
- `w600k_r50.onnx`
- `inswapper_128.onnx`
- `emap.bin`

`emap.bin` format is 8-byte little-endian header (`int32 rows`, `int32 cols`) followed by float32 matrix data. V0.9 requires 512 x 512.

The importer rejects path traversal, missing/zero-byte artifacts, duplicate required entries, EMAP wrong dimensions, and archives over the configured unpacked-size ceiling. Models are copied into app-private storage; replacing a pack is atomic.

## Static swap data flow
1. Load source and target images with bounded decoding.
2. Detect faces using 640x640 SCRFD preprocessing and 9-output landmark-capable contract.
3. Select the largest source face and largest target face.
4. Align source face to the 112x112 ArcFace template.
5. Extract and L2-normalize the 512D source embedding.
6. Align target face to the 128x128 INSwapper template.
7. Map source identity through EMAP and L2-normalize again.
8. Run INSwapper-128.
9. Invert the target alignment transform and paste the swapped face back with a deterministic feathered elliptical mask.
10. Publish a new JPEG to `Pictures/FaceRE` through MediaStore.

No unchanged input may be reported as a successful swap. Source-no-face, target-no-face, model mismatch, tensor mismatch, inference failure, or output publication failure is a hard failure.

## UI
One Activity, programmatic native Android UI:

- title `Face-RE`
- model status + `导入模型包`
- `选择源人脸`
- `选择目标图片`
- two previews
- primary action `开始换脸`
- progress/status text
- result preview
- `分享结果`

The swap button is enabled only when a model pack, source image and target image are all present and the app is idle.

## Privacy
Inference is fully local. Images and identity embeddings are never uploaded. Embeddings are job-memory only and are not persisted or logged.

## Open-source attribution
Adapted algorithmic portions from `Parasaran-Python/android-face-fusion`, MIT License, Copyright (c) 2026 Parasaran Vedanarayanan. Preserve attribution in `THIRD_PARTY_NOTICES.md` and source headers where substantial code is adapted.

## Tests and gates
Pure JVM tests cover vector normalization/EMAP mapping, model-pack filename/path contract, and similarity-transform math.

GitHub Actions must:
- run `:facere:testDebugUnitTest`;
- assemble `:facere:assembleDebug`;
- verify version `0.9.0` / code `11` and launcher `com.qujindai.facere.FaceReMainActivity`;
- verify stable certificate SHA-256 `c9329387060f0259870e30b7bca6b9d9684ed86de716676962a1e75ef7f346ac`;
- verify ONNX Runtime arm64 libraries are packaged;
- fail if APK contains `librife`, `mobilei2v`, `depth-anything`, `flownet`, or any `.onnx` model asset;
- upload the installable APK.

Functional S24U acceptance requires a user-supplied compatible model pack and two real images. CI green alone means build candidate, not visual swap acceptance.