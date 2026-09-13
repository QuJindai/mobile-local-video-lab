# V0.9 Face-RE Design

## Status

Approved direction: native Android ONNX Face-RE on top of the existing V0.8.1 handset workbench.

Base branch: `feat/v080-motion-studio`
Implementation branch: `feat/v090-face-re`
Target app version: `0.9.0` / versionCode `11`
Package: `com.qujindai.localvideo`
App minimum Android: API 28
Face-RE media-publication minimum Android: API 29
Primary validation device: Samsung S24U / arm64-v8a

## Goal

Add a genuine, fully local Face-RE pipeline to Local Video Lab. The user selects one source-face image and one target image or target video. The app detects and aligns faces, extracts the source identity embedding, swaps the selected target face, stabilizes face geometry across video frames, blends the swapped face back into the target frame, preserves target-frame presentation timing and audio where present, and publishes a real image/video result. Existing RIFE Motion, Depth 3D Motion and MobileI2V behavior must remain intact.

A run is successful only when the face-swap inference path actually executed and an output media file was produced. Face-RE must fail closed if the model pack, target face, runtime, frame decode, inference, blend, encode or remux step fails. It must never copy the input to the output and report success.

## Non-goals for V0.9

- Real-time camera face replacement.
- Multi-source identity morphing.
- Training or fine-tuning a face-swap model on the phone.
- Cloud execution or server fallback.
- Shipping third-party pretrained face-swap weights inside the public repository or APK.
- Face recognition/search against a database of people.
- Face enhancement/upscaling after swap; this can be added later as an optional stage.

## Architecture

V0.9 extends the existing local-video workbench rather than introducing a second app or service.

The execution path is:

`source face image -> source detection/alignment -> source embedding -> target media decode -> target detection/tracking -> target alignment -> swapper inference -> mask/blend -> frame stabilization -> timestamped output encode -> optional audio remux -> MediaStore publish -> ResultHistoryStore`

ONNX Runtime remains the inference runtime. Existing `ResultHistoryStore`, `ResultRecord`, worker-thread execution, progress reporting, diagnostics and stable signing identity are reused. Existing `Yuv420Frame` and `Yuv420PlaneWriter` are reused by the new timestamp-preserving video encoder.

The Face-RE subsystem is split into focused units so model parsing, face geometry, frame tracking, inference and media I/O can be tested independently.

## Model-pack contract

V0.9 reuses `local-video-model-pack-v1`. No second archive format is introduced.

A Face-RE pack has:

```properties
format=local-video-model-pack-v1
id=face-re-inswapper128
backend=face-re
version=1.0.0
license.code=MIT
license.weights=USER-SUPPLIED
min.ram.mb=6144
recommended.ram.mb=8192
files=detector.onnx,recognizer.onnx,swapper.onnx,swapper_emap.f32
sha256.detector.onnx=<64 hex chars>
sha256.recognizer.onnx=<64 hex chars>
sha256.swapper.onnx=<64 hex chars>
sha256.swapper_emap.f32=<64 hex chars>
```

`ModelPackManifest` gains `isFaceRe()`. `ModelPackStore` gains an independent Face-RE slot and must continue to store MobileI2V separately. Activating or removing one backend must never mutate the other backend's active pack.

The initial executable contract is compatible with the common SCRFD + ArcFace + InSwapper-128 graph family:

- `detector.onnx`: face bounding boxes plus 5 landmarks.
- `recognizer.onnx`: 112x112 aligned RGB face to normalized identity embedding.
- `swapper.onnx`: aligned target face plus mapped source identity embedding to swapped aligned face.
- `swapper_emap.f32`: little-endian float32 embedding-map matrix extracted from the swapper model offline by the supplied pack-builder tool.

The Android runtime does not parse ONNX protobuf initializers. `tools/facere/build_model_pack.py` extracts the swapper embedding map on a desktop/CI Python environment and writes deterministic pack metadata plus SHA-256 values. This keeps the Android dependency surface small and makes the runtime model contract explicit.

The repository and APK do not bundle InsightFace pretrained weights. InsightFace code is MIT-licensed, while its pretrained recognition and face-swap model weights have separate usage/licensing terms; InSwapper commercial use requires appropriate model licensing. Therefore V0.9 accepts user-supplied model packs and records `license.weights` in diagnostics and model details instead of silently redistributing weights.

## Core components

### `FaceReModelPack.java`

Validates Face-RE-specific required artifacts and exposes canonical `File` handles. It rejects missing files, zero-byte files and the wrong backend.

### `FaceReModelProbe.java`

Opens detector, recognizer and swapper sessions and validates required input/output ranks, element types and compatible embedding dimensions. READY state requires this probe to pass; file presence alone is insufficient.

### `FaceReFace.java`

Immutable value object containing bounding box, five landmarks and detector confidence. It contains no identity embedding.

### `FaceReDetector.java`

Owns the detector ONNX session and image preprocessing/postprocessing. Output is a list of `FaceReFace` values. Non-max suppression is deterministic.

Default target selection is the largest detected face by bounding-box area. The API also supports selection by normalized tap coordinate so the UI can select one face explicitly without changing inference code.

### `FaceReGeometry.java`

Provides 5-point similarity alignment to the 112x112 ArcFace template and inverse transform back into the source frame. Matrix math is pure Java and unit-testable without Android inference.

### `FaceReEmbedder.java`

Runs the recognizer once for the source face and L2-normalizes the embedding. A source image with no detected face or an invalid/near-zero embedding aborts before target processing starts.

### `FaceReSwapper.java`

Loads `swapper_emap.f32`, validates its shape against the recognizer/swapper contract, maps the normalized source embedding, runs swapper inference for an aligned target face, and returns an aligned swapped RGB bitmap/tensor. The mapped source vector is computed once per job and reused across all target video frames.

### `FaceReTracker.java`

Maintains one target track across video frames. Matching uses a weighted cost of IoU, landmark-center distance and scale delta. The tracker smooths bounding box and landmarks with exponential smoothing.

Rules:

- Initial frame: choose the largest face unless the user selected a face.
- Normal continuation: accept the best geometrically compatible match above tracking thresholds.
- Short miss: retain prior geometry for up to 3 consecutive frames only if frame-to-frame motion remains bounded; do not hallucinate a new identity.
- Longer miss or ambiguous identity switch: keep the original frame unchanged for that frame and continue detection on following frames; record the miss in diagnostics.
- Never jump automatically to a different person solely because that face becomes larger.

### `FaceReBlender.java`

Warps the swapped aligned face back to the target frame and composites it with a soft elliptical/landmark-derived mask. The mask is feathered and constrained inside the detected face area. V0.9 uses deterministic alpha blending; Poisson blending is not required.

### `FaceReImageEngine.java`

Runs a complete source-image + target-image job and returns a published image result. It is the fastest functional acceptance path and shares detector/embedder/swapper/blender code with video.

### `FaceReVideoDecoder.java`

Uses Android `MediaExtractor` + `MediaCodec` to decode the target video in presentation-time order. Each decoded frame carries its original `presentationTimeUs`, width, height and rotation-normalized bitmap/RGB data. It must reject unsupported decoder output instead of returning malformed frames.

The decoder records source width, height, rotation, duration and whether an audio track exists. Variable-frame-rate input is permitted because per-frame presentation timestamps are retained.

### `FaceReTimeline.java`

Pure-Java utility that validates and normalizes decoded presentation timestamps to a zero-based strictly nondecreasing sequence suitable for `MediaCodec`. It preserves source inter-frame timing deltas and provides the final video duration for audio trimming.

### `TimestampedMp4Encoder.java`

A Face-RE-specific H.264 MediaCodec encoder using the existing `Yuv420Frame` and `Yuv420PlaneWriter`. Unlike existing `Mp4Encoder`, it queues each frame with the `presentationTimeUs` supplied by `FaceReTimeline` instead of deriving timestamps from a fixed FPS. Existing RIFE/Depth/MobileI2V encoder behavior is not changed.

### `FaceReVideoEngine.java`

Creates one Face-RE job workspace in app cache, extracts source identity once, decodes the target video sequentially, runs detection/tracking/swap/blend per frame, writes numbered PNG frame files plus the normalized frame timestamps, then calls `TimestampedMp4Encoder`.

The first V0.9 video implementation is intentionally sequential and single-worker. It prioritizes deterministic behavior and bounded memory over maximum throughput. No full-video frame list of Bitmaps may be held in RAM; completed frame Bitmaps are recycled after writing.

### `AudioRemuxer.java`

After Face-RE creates the timestamp-preserving H.264 video-only MP4, `AudioRemuxer` uses `MediaExtractor` and `MediaMuxer` to copy the generated video track and the original target audio track into a final MP4 without audio re-encoding. If the target has no audio, the video-only MP4 is valid. If the target has an audio track but remux fails, the run fails rather than silently dropping audio.

Audio samples before the normalized video start are shifted to zero along with the video. Audio samples beyond the final video presentation timestamp/duration are not written.

### `FaceReMediaPublisher.java`

Publishes final Face-RE media without changing the existing video-only `MediaStorePublisher` contract.

- Images use `MediaStore.Images` with an image MIME type and `Pictures/LocalVideoLab`.
- Videos use `MediaStore.Video` with `video/mp4` and `Movies/LocalVideoLab`.
- Pending entries are deleted on copy failure.
- Face-RE generation is blocked on API 28 because the existing scoped-storage publication pattern requires API 29; the rest of the app keeps its existing minSdk 28 declaration.

### `FaceReDiagnostics.java`

Records model-pack identity/hashes, runtime provider, frame count, swapped-frame count, unchanged-frame count, detector misses, tracker re-acquisitions, average detection/inference/blend latency, peak Java/native memory observations, output size, audio remux status and total elapsed time.

No face embedding vectors are written to diagnostics, logs or persistent storage.

## UI integration

`BackendRouter.Backend` gains `FACE_RE` and the backend spinner gains:

`Face-RE · 本地换脸 / ONNX Runtime`

Face-RE input semantics are explicit and do not overload the existing RIFE labels:

- Source: `源人脸` — image only.
- Target: `目标图片/视频` — image or video.
- Face selector: default `最大目标脸`; if multiple faces are detected on an image preview, tapping a face stores a normalized target point for selection. Video V0.9 uses the first decoded frame for target selection and tracker continuation.

When Face-RE is selected, RIFE/depth/motion/frame-count controls are hidden or disabled and Face-RE controls are shown. Existing backends retain their current controls and behavior.

The primary action text becomes `开始换脸` for Face-RE. The result card, share button and history list are reused. Result recipe text is `Face-RE · <pack id> · swapped X/Y frames`.

The model panel shows a separate Face-RE pack section with import/remove controls, version, source metadata, code/weights license text and READY/NOT READY state. A Face-RE pack is READY only when the manifest is valid, all SHA-256 hashes pass, all required model files exist, ONNX Runtime loads, the Face-RE model probe passes, RAM meets the pack minimum and Android is API 29 or newer.

## Routing and state

`BackendRouter.Backend` gains `FACE_RE`. Face-RE blockers are:

- `FACE_RE_PACK_MISSING`
- `FACE_RE_PACK_INVALID`
- `FACE_RE_RUNTIME_UNAVAILABLE`
- `FACE_RE_INSUFFICIENT_RAM`
- `FACE_RE_UNSUPPORTED_ANDROID`

The minimum RAM gate is read from the active pack manifest, with 6144 MB as the initial pack floor. S24U is expected to pass this gate.

`MainActivityV05.startGeneration()` delegates Face-RE to `FaceReImageEngine` or `FaceReVideoEngine` based on target MIME type. Face-RE never routes through `RifeEngine` or `MobileI2VGpuEngine`.

## Media output

Image target:

- Preserve target width/height.
- Publish a new result image through `FaceReMediaPublisher`.
- Never overwrite the input URI.

Video target:

- Preserve target display dimensions.
- Preserve decoded source frame presentation-time deltas through `TimestampedMp4Encoder`; do not synthesize timestamps from nominal FPS.
- Restore original audio through `AudioRemuxer` when audio exists.
- Publish to `Movies/LocalVideoLab` only after video and audio validation gates succeed.
- Temporary frames and intermediate MP4s are deleted in `finally` blocks.

## Error handling and fail-closed rules

A Face-RE run is rejected before execution when either input is missing, source MIME is not image, target MIME is not image/video, Android is below API 29, the active pack is unavailable/invalid, model contract probing fails, or RAM is below the pack minimum.

During execution:

- No source face -> fail with `source face not detected`.
- Target image no face -> fail with `target face not detected`.
- Video with no detected target face in any frame -> fail; do not publish an unchanged copy.
- Individual video frame tracking miss -> keep that original frame, increment `unchangedFrames`, continue.
- If `swappedFrames == 0` -> fail and delete output.
- Invalid or decreasing decoded timestamps -> fail before encode; do not silently reorder frames.
- Any ONNX tensor shape/type mismatch -> fail and expose the actual model/input name in diagnostics.
- Encoder/muxer failure -> fail and delete intermediates.
- Existing result history is written only after MediaStore publication succeeds.

## Privacy and persistence

All Face-RE inference is local. No source images, target frames, embeddings or generated media are uploaded by the Face-RE code path.

Source identity embeddings live in job memory only and are released when the job finishes. They are never saved in SharedPreferences, model-pack storage, diagnostics or result history.

The active model-pack path may persist, matching the existing MobileI2V behavior.

## Tests

### Pure JVM tests

Add tests for:

- `ModelPackManifest.isFaceRe()` and independent model-store routing.
- Required Face-RE artifacts and backend rejection.
- model-probe contract helpers for accepted/rejected tensor shapes.
- 5-point similarity transform/inverse transform round trip.
- L2 embedding normalization and zero-vector rejection.
- embedding-map dimensions and deterministic mapping.
- NMS ordering.
- largest-face selection and tap-coordinate selection.
- tracker continuity, smoothing, 3-frame miss window and prevention of person jumping.
- blend-mask bounds.
- timestamp normalization preserving source deltas and rejecting decreasing PTS.
- backend routing, Android-version and RAM gates.
- result success guard requiring `swappedFrames > 0`.

### Host pack-builder tests

`tools/facere/test_build_model_pack.py` builds a deterministic synthetic ONNX fixture, extracts an embedding map, produces a pack and verifies hashes/manifest ordering. Tests must not require proprietary or third-party production weights.

### Android/CI gates

The existing Android workflow is extended to:

- run all current V0.8.1 tests unchanged;
- run new Face-RE JVM tests;
- test the pack builder;
- assemble version `0.9.0`, versionCode `11`;
- verify the stable signing certificate is unchanged;
- verify `onnxruntime` libraries remain packaged for arm64-v8a and meet existing alignment gates;
- verify Face-RE classes and manifest support are in the APK;
- verify no third-party face model `.onnx` weights are accidentally bundled under assets;
- upload a S24U APK artifact.

### S24U handset acceptance

Handset acceptance requires all of the following:

1. Upgrade-install over V0.8.1 without deleting app data.
2. Existing RIFE Motion generation still succeeds.
3. Existing Depth 3D generation still succeeds.
4. Import a Face-RE model pack and obtain READY state with visible pack identity/license metadata.
5. Static image: one source face + one target image produces a visibly swapped face and a new saved result.
6. Video: one source face + a 10-second target video produces a playable swapped MP4.
7. For a target video with audio, output audio is present and duration-aligned.
8. For a variable-frame-rate sample, output frame-time deltas remain aligned with source PTS within normal MediaCodec timestamp granularity.
9. No obvious per-frame face jumping/flicker under ordinary pose changes; tracker diagnostics show continuous tracking rather than random target switching.
10. Remove/rename one required pack artifact and verify generation is blocked rather than silently falling back.
11. Run with a target containing no face and verify no output is falsely reported as successful.
12. Export diagnostics and confirm model identity, frame totals, swapped/unchanged counts, timing and remux status are present, while raw embeddings are absent.

## Release gate

V0.9 may be called `Face-RE functional` only after CI passes and the S24U image and 10-second video acceptance runs pass with a real user-supplied model pack.

Before handset acceptance, the APK must be labeled a candidate even if compilation and unit tests are green.

The V0.9 branch must not be merged into `main` until the existing V0.8.1 regression gates plus all Face-RE gates pass.