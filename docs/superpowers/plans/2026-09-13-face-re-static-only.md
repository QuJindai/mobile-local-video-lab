# Face-RE Static-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single-purpose S24U Android APK that performs local source-image to target-image face replacement and contains no RIFE/Depth/MobileI2V runtime or model assets.

**Architecture:** Add a new `:facere` application module and make it the only included Gradle module on the feature branch. The module uses ONNX Runtime 1.29.0 with a user-imported four-file model ZIP, a focused SCRFD→ArcFace→INSwapper pipeline, native Android bitmap alignment/blending, and MediaStore JPEG publication.

**Tech Stack:** Java 17, Android SDK 35/minSdk 29, ONNX Runtime Android 1.29.0, JUnit 4, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-face-re-static-only-design.md`

## Global Constraints
- Version `0.9.0`, versionCode `11`.
- Application ID stays `com.qujindai.localvideo`.
- arm64-v8a only.
- Stable signing certificate SHA-256 stays `c9329387060f0259870e30b7bca6b9d9684ed86de716676962a1e75ef7f346ac`.
- No production face-model weights in repository or APK.
- APK must not package RIFE, Depth Anything, MobileI2V, or motion runtime assets.
- Static images only; no video code path.

---

### Task 1: Create isolated Face-RE app module and RED tests

**Files:**
- Modify: `settings.gradle`
- Create: `facere/build.gradle`
- Create: `facere/src/main/AndroidManifest.xml`
- Create: `facere/src/test/java/com/qujindai/facere/VectorMathTest.java`
- Create: `facere/src/test/java/com/qujindai/facere/FacePackContractTest.java`
- Create: `facere/src/test/java/com/qujindai/facere/SimilarityMathTest.java`
- Create: `.github/workflows/face-re-apk.yml`

**Interfaces:**
- Tests require `VectorMath.normalize(float[])`, `VectorMath.mapAndNormalize(float[], float[][])`, `FacePackContract.isSafeEntry(String)`, `FacePackContract.hasRequired(Set<String>)`, and `SimilarityMath.estimate(float[][], float[][])`.

- [ ] Write tests before production classes.
- [ ] Push RED state and confirm Actions fails because the required production classes do not yet exist.
- [ ] Do not create runtime implementation until the failure is observed.

### Task 2: Implement pure math and pack contract GREEN

**Files:**
- Create: `facere/src/main/java/com/qujindai/facere/VectorMath.java`
- Create: `facere/src/main/java/com/qujindai/facere/FacePackContract.java`
- Create: `facere/src/main/java/com/qujindai/facere/SimilarityMath.java`

**Interfaces:**
- `normalize` rejects null/empty/near-zero/non-finite vectors.
- `mapAndNormalize` requires square matrix dimension equal to embedding length and computes row-vector × matrix before normalization.
- `estimate` returns six affine values `[a,b,c,d,e,f]` mapping source points to destination points.

- [ ] Implement only behavior required by tests.
- [ ] Run unit-test CI until GREEN.

### Task 3: Implement secure model ZIP import

**Files:**
- Create: `facere/src/main/java/com/qujindai/facere/FaceModelStore.java`
- Extend: `FacePackContractTest.java`

**Interfaces:**
- Required names: `det_10g.onnx`, `w600k_r50.onnx`, `inswapper_128.onnx`, `emap.bin`.
- `FaceModelStore.importZip(Context, Uri, Progress)` extracts to a staging directory, validates required files and 512x512 EMAP, then atomically replaces active pack.
- `active(Context)` returns a `FaceModelFiles` value or null.

- [ ] Add failing tests for entry/path validation and required-file completeness.
- [ ] Implement importer and active-pack resolution.

### Task 4: Implement SCRFD detector and source embedding

**Files:**
- Create: `facere/src/main/java/com/qujindai/facere/OrtSessions.java`
- Create: `facere/src/main/java/com/qujindai/facere/DetectedFace.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceDetector.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceAligner.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceEmbedder.java`

**Interfaces:**
- Detector supports the landmark-capable 9-output SCRFD contract at strides 8/16/32, confidence fallback 0.5→0.3, NMS 0.4.
- `DetectedFace.largest(List<DetectedFace>)` selects by bounding-box area.
- `FaceAligner.align112/align128` returns bitmap plus forward transform.
- Embedder returns a normalized 512D identity vector.

- [ ] Adapt algorithmic behavior from MIT `android-face-fusion`, replacing its downloader with `FaceModelStore` file handles.
- [ ] Keep substantial-source attribution in headers/notices.

### Task 5: Implement INSwapper and blend

**Files:**
- Create: `facere/src/main/java/com/qujindai/facere/FaceSwapper.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceBlender.java`
- Extend: `VectorMathTest.java`

**Interfaces:**
- Swapper resolves image and embedding input names by tensor rank.
- EMAP is 512x512 little-endian float matrix loaded from imported pack.
- Target input normalization is RGB / 255.0.
- Output is converted from `[1,3,128,128]` float to ARGB bitmap.
- Blender inverse-warps swapped face and a feathered elliptical mask into target coordinates.

- [ ] Add failing EMAP mapping edge tests.
- [ ] Implement and keep tests GREEN.

### Task 6: Implement static engine, output, and UI

**Files:**
- Create: `facere/src/main/java/com/qujindai/facere/ImageLoader.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceSwapEngine.java`
- Create: `facere/src/main/java/com/qujindai/facere/ImagePublisher.java`
- Create: `facere/src/main/java/com/qujindai/facere/FaceReMainActivity.java`

**Interfaces:**
- Engine performs exactly one source→one target static swap and fails if either image has no face.
- UI imports model ZIP, selects source/target images, runs engine on one background executor, previews/publishes/shares result.
- Success state is entered only after swapped inference result is published.

### Task 7: Strip APK payload and verify candidate

**Files:**
- Modify: `THIRD_PARTY_NOTICES.md`
- Finalize: `.github/workflows/face-re-apk.yml`

**Gates:**
- `gradle :facere:testDebugUnitTest` PASS.
- `gradle :facere:assembleDebug` PASS.
- APK badging: versionName `0.9.0`, code `11`, launcher `com.qujindai.facere.FaceReMainActivity`.
- Existing stable signer PASS.
- ONNX Runtime arm64 libraries present.
- No `librife`, `mobilei2v`, `depth-anything`, `flownet`, or `.onnx` asset in APK.
- Upload artifact `face-re-v0.9.0-static-apk`.

### Task 8: Handset handoff

- Download the successful Actions artifact.
- Verify artifact/APK SHA-256 and signing metadata.
- Provide APK to user for S24U model-pack and visual static-swap acceptance.
- Do not claim visual swap acceptance until a real compatible model pack and source/target photos are tested on handset.