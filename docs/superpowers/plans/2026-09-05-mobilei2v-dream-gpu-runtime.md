# MobileI2V Dream-GPU Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MobileI2V a genuine on-device accelerated backend on SM8650, using Adreno GPU/MNN OpenCL as the first production path, Dream-derived runtime patterns, real model-pack evidence, and CPU only for diagnostics.

**Architecture:** Add a v2 accelerated model-pack contract, a pinned host exporter, an Android native MNN/OpenCL core, a Flow Euler sampler, and truthful readiness/microscope reporting. Reuse Local-Dream architectural patterns (MNN OpenCL tuning cache, mmap, staged model lifetime, QNN backend identity) while keeping its Stable Diffusion implementation distinct from MobileI2V.

**Tech Stack:** Android Java 17, NDK 27.2, C++17, MNN OpenCL, JNI, Python/PyTorch/ONNX for export, GitHub Actions, existing Android MediaCodec/RIFE export infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-05-mobilei2v-dream-gpu-runtime-design.md`

## Global Constraints

- MobileI2V source pin: `hustvl/MobileI2V@8d0a253c766b05a43ba408baf5e8f800a36be8b4`.
- Checkpoint SHA-256: `bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d`.
- Dream reference pin: `xororz/local-dream@a7666f6198412a58c6eb1eacc28828aa40c7d7ae`.
- MNN pin inherited from Dream: `3db3cc904dfea55286972b472b040ad5525aa083`.
- Production MobileI2V denoiser/VAE must not silently execute on CPU.
- RIFE must never be used to fabricate MobileI2V generation.
- `SOURCE CHECKPOINT READY` and `GENERATION READY` are separate states.
- Runtime pack target: 17 frames, 1280x720, fixed latent shape `[1,128,3,23,40]`.

---

### Task 1: Accelerated model-pack v2 contract

**Files:**
- Create: `app/src/main/java/com/qujindai/localvideo/AcceleratedPackManifest.java`
- Create: `app/src/test/java/com/qujindai/localvideo/AcceleratedPackManifestTest.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/InstalledModelPack.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/ModelPackInstaller.java`

**Interfaces:**
- Produces: `AcceleratedPackManifest.parse(InputStream)`, `isMobileI2VGpuRunnable()`, `execution`, `dreamCommit`, `mnnCommit`, `checkpointSha256`, `frames`, `width`, `height`.
- Existing v1 packs remain readable; only v2 packs can satisfy accelerated MobileI2V readiness.

- [ ] **Step 1: Write the failing parser/readiness tests**

```java
@Test public void v2GpuPackRequiresPinnedExecutionFields() throws Exception {
    String p = "format=local-video-model-pack-v2\n"
        + "id=mobilei2v-300m-gpu\nbackend=mobilei2v\nversion=0.7\n"
        + "execution=mnn-opencl\n"
        + "source.repo=hustvl/MobileI2V\n"
        + "source.commit=8d0a253c766b05a43ba408baf5e8f800a36be8b4\n"
        + "checkpoint.sha256=bc6a545302b342b87d83a4d78e9b74d47ca59fbf908fd8e13d9ecedbe1a37f2d\n"
        + "dream.source=xororz/local-dream\n"
        + "dream.commit=a7666f6198412a58c6eb1eacc28828aa40c7d7ae\n"
        + "mnn.commit=3db3cc904dfea55286972b472b040ad5525aa083\n"
        + "frames=17\nwidth=1280\nheight=720\n"
        + "files=denoiser.mnn,vae_encoder.mnn,vae_decoder.mnn,empty_prompt.f16,empty_prompt_mask.bin\n"
        + hashesForFiveFiles();
    AcceleratedPackManifest m = AcceleratedPackManifest.parse(stream(p));
    assertTrue(m.isMobileI2VGpuRunnable());
    assertEquals("mnn-opencl", m.execution);
}

@Test public void rawOrLegacyPackDoesNotBecomeGpuRunnable() throws Exception {
    assertFalse(AcceleratedPackManifest.fromLegacy(legacyManifest()).isMobileI2VGpuRunnable());
}
```

- [ ] **Step 2: Run test and verify RED**

Run: `gradle :app:testDebugUnitTest --tests '*AcceleratedPackManifestTest' --stacktrace`
Expected: compile failure because `AcceleratedPackManifest` does not exist.

- [ ] **Step 3: Implement strict v2 parsing and pin validation**

Implement exact token/SHA/path validation and reject missing accelerated artifacts. Do not accept `execution=cpu` as runnable.

- [ ] **Step 4: Run parser and full JVM tests**

Run: `gradle :app:testDebugUnitTest --stacktrace`
Expected: PASS.

- [ ] **Step 5: Commit**

`git commit -am "feat: add accelerated MobileI2V pack contract"`

---

### Task 2: Truthful accelerated runtime readiness and microscope model

**Files:**
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VRuntimeProbe.java`
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VMicroscope.java`
- Create: `app/src/test/java/com/qujindai/localvideo/MobileI2VRuntimeProbeTest.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/BackendRouter.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/OnnxRuntimeFoundation.java`

**Interfaces:**

```java
final class MobileI2VRuntimeProbe {
  enum Backend { MNN_OPENCL, QNN_HTP, CPU_DIAGNOSTIC, NONE }
  static Decision decide(boolean packValid, boolean nativeLoaded,
      boolean openClReady, boolean qnnReady, long ramMb);
}
```

- [ ] **Step 1: Write failing readiness tests**

```java
@Test public void rawCheckpointNeverMakesMobileReady() {
  Decision d = MobileI2VRuntimeProbe.decide(false, true, true, false, 11000);
  assertFalse(d.ready);
}
@Test public void openClPackMakesGpuReady() {
  Decision d = MobileI2VRuntimeProbe.decide(true, true, true, false, 11000);
  assertTrue(d.ready);
  assertEquals(Backend.MNN_OPENCL, d.backend);
}
@Test public void noSilentCpuFallback() {
  Decision d = MobileI2VRuntimeProbe.decide(true, true, false, false, 11000);
  assertFalse(d.ready);
  assertEquals(Backend.NONE, d.backend);
}
```

- [ ] **Step 2: Verify RED**

Run: `gradle :app:testDebugUnitTest --tests '*MobileI2VRuntimeProbeTest' --stacktrace`
Expected: FAIL because runtime probe is missing.

- [ ] **Step 3: Implement readiness and microscope record types**

`MobileI2VMicroscope` must contain pack/source/runtime/backend/step/timing/memory/thermal fields and `boolean cpuProductionPathUsed`.

- [ ] **Step 4: Run all JVM tests**

Run: `gradle :app:testDebugUnitTest --stacktrace`
Expected: PASS.

- [ ] **Step 5: Commit**

`git commit -am "feat: add truthful MobileI2V accelerator readiness"`

---

### Task 3: Dream-derived MNN/OpenCL native core

**Files:**
- Create: `app/src/main/cpp/mobilei2v/CMakeLists.txt`
- Create: `app/src/main/cpp/mobilei2v/mnn_runtime.hpp`
- Create: `app/src/main/cpp/mobilei2v/mobilei2v_jni.cpp`
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VGpuNative.java`
- Create: `scripts/prepare_mobilei2v_gpu_runtime.sh`
- Modify: `app/build.gradle`
- Test: CI source/build gates in `.github/workflows/android-apk.yml`

**Interfaces:**

```java
final class MobileI2VGpuNative {
  static native String probe(String modelDir);
  static native long load(String modelDir);
  static native int runDenoiser(long handle, float[] latent, float timestep,
      float[] prompt, byte[] mask, float flowScore, float[] output);
  static native void release(long handle);
}
```

- [ ] **Step 1: Add a failing CI/source contract**

CI must check the pinned Dream/MNN commits, `MNN_OPENCL=ON`, `MNN_FORWARD_OPENCL`, `MNN_GPU_MEMORY_BUFFER`, `MNN_GPU_TUNING_FAST`, and JNI library presence.

- [ ] **Step 2: Verify RED in Actions**

Expected: `Verify MobileI2V GPU runtime contract` fails before native core exists.

- [ ] **Step 3: Implement pinned native build**

`scripts/prepare_mobilei2v_gpu_runtime.sh` clones MNN at `3db3cc...`, configures Android arm64 with OpenCL enabled, and builds `libmobilei2v_gpu.so`. `mnn_runtime.hpp` mirrors the validated Dream ideas: mmap model load, OpenCL low precision, fast tuning, per-stage cache file, and no implicit CPU fallback.

- [ ] **Step 4: Build arm64 core**

Run in CI: `bash scripts/prepare_mobilei2v_gpu_runtime.sh`
Expected: `app/src/main/jniLibs/arm64-v8a/libmobilei2v_gpu.so` plus required MNN/OpenCL libraries.

- [ ] **Step 5: Verify 16-KB ELF alignment**

Use NDK `llvm-readelf -lW` and require every `LOAD` alignment >= `0x4000`.

- [ ] **Step 6: Commit**

`git commit -am "feat: add Dream-derived MNN OpenCL GPU core"`

---

### Task 4: Real MobileI2V exporter and deterministic GPU pack

**Files:**
- Create: `tools/mobilei2v/export_mobilei2v_gpu.py`
- Create: `tools/mobilei2v/build_gpu_pack.py`
- Create: `tools/mobilei2v/test_gpu_pack.py`
- Create: `.github/workflows/mobilei2v-gpu-pack.yml`

**Interfaces:**
- Export fixed denoiser input: CFG batch 2, latent `[2,128,3,23,40]`, timestep `[2]`, prompt `[2,1,300,896]`, condition mask `[1,2760]`, flow score `[2]`.
- Pack output: `dist/mobilei2v-300m-mnn-opencl-v0.7.mlvpkg`.

- [ ] **Step 1: Write failing deterministic pack tests**

Tests must reject a pack without `denoiser.mnn`, both VAE artifacts, empty prompt artifacts, pins, or hashes; rebuilds from identical inputs must have identical SHA-256.

- [ ] **Step 2: Verify RED**

Run: `python3 tools/mobilei2v/test_gpu_pack.py`
Expected: FAIL because GPU pack builder does not exist.

- [ ] **Step 3: Implement exporter**

The script must clone/import the pinned upstream, verify the raw checkpoint SHA, build `Mobiledit_300M_P1_D16`, load weights, precompute empty-prompt features, export fixed-shape ONNX, run ONNX Runtime parity, convert to MNN, and run MNN parity. Any parity failure exits non-zero and publishes no pack.

- [ ] **Step 4: Export upstream-compatible VAE path**

Export encoder and decoder required by the pinned upstream. Add Turbo-VAED only if latent-shape parity with MobileI2V passes; record `vae.impl=turbo-vaed` only after that gate.

- [ ] **Step 5: Build deterministic pack and verify**

Run: `python3 tools/mobilei2v/build_gpu_pack.py ...` then `python3 tools/mobilei2v/test_gpu_pack.py`.
Expected: PASS and deterministic SHA.

- [ ] **Step 6: Publish Actions artifact only after parity**

Artifact name: `mobilei2v-300m-mnn-opencl-v0.7-pack`.

- [ ] **Step 7: Commit**

`git commit -am "feat: export real MobileI2V GPU model pack"`

---

### Task 5: Flow Euler sampler and guide-latent lock

**Files:**
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VFlowEuler.java`
- Create: `app/src/test/java/com/qujindai/localvideo/MobileI2VFlowEulerTest.java`

**Interfaces:**

```java
interface Denoiser { void run(float[] latentCfg2, float t, float[] outCfg2); }
static float[] sample(float[] initial, float[] guideFirstSlice,
    int steps, float cfgScale, Denoiser denoiser, Progress progress);
```

- [ ] **Step 1: Write failing scheduler tests**

Tests verify monotonically progressing timesteps, CFG math, and exact guide-first-slice replacement after every step.

- [ ] **Step 2: Verify RED**

Run: `gradle :app:testDebugUnitTest --tests '*MobileI2VFlowEulerTest' --stacktrace`.

- [ ] **Step 3: Implement scheduler without PyTorch**

Match upstream `FlowMatchEulerDiscreteScheduler(shift=3.0)` semantics and the guide-latent lock. No RIFE calls are permitted in this class.

- [ ] **Step 4: Run tests**

Run: `gradle :app:testDebugUnitTest --stacktrace`.
Expected: PASS.

- [ ] **Step 5: Commit**

`git commit -am "feat: add MobileI2V Flow Euler sampler"`

---

### Task 6: End-to-end MobileI2V GPU engine and MP4 path

**Files:**
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VGpuEngine.java`
- Create: `app/src/main/java/com/qujindai/localvideo/MobileI2VFrameSink.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/Mp4Encoder.java`
- Modify: `app/src/main/java/com/qujindai/localvideo/MainActivityV05.java`

**Interfaces:**

```java
Result generate(Uri image, InstalledModelPack pack, int steps, float flowScore,
    ProgressCallback progress);
```

- [ ] **Step 1: Write engine-state tests**

Tests assert that generation is rejected for raw `.pth`, legacy pack, missing GPU probe, or any CPU production backend.

- [ ] **Step 2: Verify RED**

Run targeted JVM tests and confirm missing engine failure.

- [ ] **Step 3: Implement staged accelerated pipeline**

Order: image preprocess -> GPU VAE encode -> Flow Euler GPU denoise -> GPU VAE decode -> incremental frame sink -> MediaCodec MP4. Release each heavy stage before loading the next when safe.

- [ ] **Step 4: Wire UI**

For MobileI2V selection, `开始生成` becomes enabled only when `MobileI2VRuntimeProbe` is READY. Do not retain the existing hard-coded `!mobile` disable. Show `MobileI2V · Adreno GPU · MNN OpenCL` or the exact blocker.

- [ ] **Step 5: Add microscope output**

Display actual backend, commits, hashes, cache status, timings, memory, thermal, and CPU production-path flag.

- [ ] **Step 6: Run full tests**

Run: `gradle :app:testDebugUnitTest --stacktrace`.
Expected: PASS.

- [ ] **Step 7: Commit**

`git commit -am "feat: wire end-to-end MobileI2V GPU generation"`

---

### Task 7: V0.7 CI/APK gate and integrated handset acceptance bundle

**Files:**
- Modify: `app/build.gradle`
- Modify: `.github/workflows/android-apk.yml`
- Create: `docs/V0.7_S24U_GPU_ACCEPTANCE.md`

- [ ] **Step 1: Bump APK metadata**

Set `versionCode 7`, `versionName '0.7.0'`, UI badge `V0.7`.

- [ ] **Step 2: Strengthen CI**

CI order: model-pack tests -> accelerated runtime contract -> GPU native build -> JVM tests -> RIFE build -> APK build -> v0.7 APK content/signing/16KB checks -> artifact upload.

- [ ] **Step 3: Reject dishonest builds**

CI greps/tests must fail if MobileI2V UI says READY based only on checkpoint presence, or if engine routes MobileI2V to RIFE/CPU.

- [ ] **Step 4: Build APK**

Run: `gradle :app:assembleDebug --stacktrace`.
Expected: exit 0.

- [ ] **Step 5: Verify APK**

Verify version 0.7.0, arm64, stable signing certificate, `libmobilei2v_gpu.so`, MNN/OpenCL libs, RIFE libs, and all arm64 ELF 16-KB alignment.

- [ ] **Step 6: Publish one installable artifact**

Artifact: `mobile-local-video-v0.7-gpu-apk`.

- [ ] **Step 7: Final integrated S24U acceptance document**

One run only: install APK + GPU pack, generate a 17-frame video, airplane-mode repeat, save/share MP4, export microscope diagnostics showing non-CPU accelerated backend.

- [ ] **Step 8: Commit**

`git commit -am "ci: gate Local Video Lab V0.7 GPU runtime"`
