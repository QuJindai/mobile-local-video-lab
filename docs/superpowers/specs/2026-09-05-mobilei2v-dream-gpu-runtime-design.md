# MobileI2V Dream-GPU Runtime Design

## Goal
Turn the current MobileI2V model-download workbench into a genuine on-device MobileI2V generation path on the user's Snapdragon 8 Gen 3 / SM8650 handset, with GPU/HTP acceleration as the normal path and CPU limited to diagnostics. Reuse the proven architectural outcomes of `xororz/local-dream` without pretending its Stable Diffusion model runtime is itself MobileI2V.

## Ground truth and constraints

- MobileI2V upstream is pinned to `hustvl/MobileI2V@8d0a253c766b05a43ba408baf5e8f800a36be8b4` and `hybrid_371.pth` SHA-256 `bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d`.
- Official MobileI2V inference is PyTorch/CUDA. It produces a 17-frame video from one image, uses a 270M `Mobiledit_300M_P1_D16` denoiser, Flow Euler sampling, a video VAE, and Qwen text features. The public repo does not ship an Android runtime.
- The current app already downloads the original checkpoint from Hugging Face or HF-Mirror with shared SHA-256 and resumable storage. The raw `.pth` is not Android executable and must never be reported as a ready model pack.
- `xororz/local-dream` is pinned at `a7666f6198412a58c6eb1eacc28828aa40c7d7ae`. Its useful outcomes are architecture patterns: MNN OpenCL on Android, OpenCL tuning cache, mmap model loading, staged/low-RAM model lifetime, QNN/HTP runtime loading, backend identity reporting, and native local execution. We do not relabel Local-Dream Stable Diffusion as MobileI2V.
- Local-Dream's MNN dependency is pinned at submodule commit `3db3cc904dfea55286972b472b040ad5525aa083` and is built with `MNN_OPENCL=ON`. This is the first execution target for MobileI2V because it maps to the Adreno GPU without requiring a device-specific QNN context binary.
- The app must not use CPU as the normal MobileI2V path. CPU is permitted only for capability probes, metadata parsing, lightweight preprocessing, and explicit diagnostic fallback.
- Existing RIFE and Depth3D backends remain independent and must never be used to fabricate a MobileI2V success.

## Architecture

### 1. Runnable model pack v2

Introduce `local-video-model-pack-v2` for accelerated MobileI2V artifacts. A runnable pack contains:

- `runtime.properties`
- `denoiser.mnn`
- `vae_encoder.mnn`
- `vae_decoder.mnn` (prefer Turbo-VAED-compatible decoder when an export passes parity)
- `empty_prompt.f16` and `empty_prompt_mask.bin` so the handset does not need Qwen2-0.5B for the current no-prompt UI
- per-file SHA-256 values in the pack manifest

Required manifest fields:

- `backend=mobilei2v`
- `execution=mnn-opencl`
- `source.repo=hustvl/MobileI2V`
- `source.commit=8d0a253c766b05a43ba408baf5e8f800a36be8b4`
- `checkpoint.sha256=bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d`
- `dream.source=xororz/local-dream`
- `dream.commit=a7666f6198412a58c6eb1eacc28828aa40c7d7ae`
- `mnn.commit=3db3cc904dfea55286972b472b040ad5525aa083`
- `frames=17`, `width=1280`, `height=720`

The legacy v1 pack remains readable for compatibility but does not make MobileI2V READY unless it declares an executable accelerated runtime.

### 2. Export pipeline

A host-side exporter must build real runtime artifacts from the pinned upstream model rather than inventing model files.

The exporter:

1. Clones MobileI2V at the pinned commit.
2. Verifies `hybrid_371.pth` against the pinned SHA-256.
3. Builds the fixed 720p latent shape used by upstream: one generation latent `[1,128,3,23,40]`; classifier-free guidance evaluates a batch of 2.
4. Freezes the current app's empty-prompt path by precomputing the upstream Qwen condition/uncondition embeddings and attention mask on the host.
5. Exports the Mobiledit denoiser with fixed shapes, runs ONNX host parity, converts it to MNN, and runs MNN host parity before accepting it.
6. Exports the VAE encoder and decoder. Prefer Turbo-VAED decoder only if its latent contract is proven compatible and parity passes; otherwise retain the upstream-compatible decoder and mark Turbo-VAED as not yet active.
7. Builds a deterministic `.mlvpkg`, writes hashes, and publishes the pack only when parity and manifest verification pass.

No pack may be published on an export failure. A failed conversion is a blocked gate, not a reason to fall back to RIFE or CPU generation.

### 3. Android GPU runtime

Add `libmobilei2v_gpu.so`, linked against MNN built from the pinned Local-Dream MNN commit with OpenCL enabled. The JNI boundary exposes:

- `probe(modelDir) -> RuntimeProbe`
- `load(modelDir) -> sessionId`
- `generate(sessionId, inputImage, steps, flowScore, callback) -> GenerationResult`
- `release(sessionId)`

Runtime behavior mirrors the useful Local-Dream patterns:

- model files loaded by mmap;
- `MNN_FORWARD_OPENCL` as the production backend;
- `MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST`;
- low precision for GPU execution;
- per-stage tuning caches stored under `<modelDir>/cache/`;
- stage lifetime kept low: VAE encoder is released before denoising; denoiser is released before VAE decode where practical;
- the runtime records the actual backend used and refuses to report GPU success when OpenCL session creation falls back or fails.

The first working runtime target is Adreno/OpenCL. QNN/HTP is a second accelerator target using the same pack/state interfaces once a valid QNN context export exists. It must be reported separately as `QNN_HTP`, never as `GPU`.

### 4. MobileI2V sampler

Implement the upstream Flow Euler loop without PyTorch:

- fixed scheduler compatible with upstream `FlowMatchEulerDiscreteScheduler(shift=3.0)`;
- CFG with the frozen empty prompt condition/uncondition pair;
- first latent temporal slice is locked to the guide-image latent after every step, matching upstream behavior;
- default steps follow upstream semantics; app may expose a fast/recommended choice only when both are tested;
- no CPU denoiser path is selected automatically.

### 5. Video decode and export

The GPU runtime returns decoded frames incrementally to avoid retaining an entire 17x720p float tensor in Java memory. Existing Android MP4 encoding remains the final container writer. The generation result records encode, denoise, decode, and MP4 stages separately.

### 6. UI and microscope

The MobileI2V card must display a two-axis identity:

- Model: `MobileI2V 0.27B`
- Execution: `Adreno GPU · MNN OpenCL`, `QNN HTP`, or `NOT READY`

Microscope data must include:

- pack ID/version and source commit
- checkpoint SHA prefix
- Dream baseline commit and MNN commit
- requested backend and actual backend
- OpenCL availability and tuning-cache state
- VAE implementation (`upstream` or `Turbo-VAED`)
- steps, 17-frame/720p contract
- encode / denoise / decode / MP4 milliseconds
- Java/native memory before/peak/after
- thermal status before/after
- explicit CPU-use flag; a production MobileI2V run fails the acceptance gate when denoising or VAE execution is CPU

### 7. Readiness policy

`MobileI2V READY` requires all of the following:

1. valid v2 runtime pack installed;
2. manifest/source/checkpoint hashes match pins;
3. native GPU core loads;
4. OpenCL backend probe succeeds;
5. denoiser + VAE runtime smoke executes on the requested accelerated backend;
6. device RAM >= 8 GB;
7. no backend substitution.

A raw downloaded `.pth` alone is `SOURCE CHECKPOINT READY`, not `GENERATION READY`.

## Acceptance

Cloud/CI acceptance:

- all JVM tests pass;
- pack-v2 parser/security tests pass;
- exporter source/checkpoint pin checks pass;
- model export parity gates pass before a real runtime pack is published;
- Android arm64 GPU native core builds with OpenCL enabled;
- APK contains the GPU core and required runtime libraries, is 16-KB aligned, correctly signed, and reports V0.7 metadata;
- CI rejects any build that marks MobileI2V ready without accelerated-runtime evidence.

Final S24U acceptance is one integrated handset run, not micro-tests:

1. install one APK;
2. install/download the real v2 MobileI2V GPU pack;
3. select one image and generate a 17-frame video;
4. repeat once in airplane mode;
5. verify microscope reports `Adreno GPU · MNN OpenCL` (or a separately identified `QNN HTP`) and no production CPU denoise/VAE path;
6. save/share the resulting MP4 and export the single diagnostics report.
