# V0.7.1 handset test APK

The user requested returning to main development and receiving an APK for local
testing. Deliver a newly built, consistently signed test APK with the known
Android MobileI2V contract defects corrected. Preserve existing RIFE, Depth 3D,
history and sharing. Keep main frozen and PR #4 draft. No paid GPU jobs or
billing work in this task. Do not claim complete MobileI2V generation: the
denoiser model is still not numerically qualified and no GPU pack is delivered.

## Task 1: Native contract correction

Work only on app/src/main/cpp/mobilei2v/mnn_runtime.hpp and mobilei2v_jni.cpp.
Use the locally pinned MNN source /dev/shm/mobilei2v-mnn-source to check APIs.
The VAE encoder has FP32 image [1,3,720,1280] and posterior_epsilon
[1,128,1,23,40], output guide [1,128,1,23,40]. JNI nativeEncode takes
(long, float[] image, float[] epsilon, float[] guide). The denoiser has ONLY
latent [2,128,3,23,40], timestep [2], cond_mask [1,2760], flow_score [2],
output [2,128,3,23,40]. Remove all prompt/text-mask files, reads and ports.
Only FP32 runtime ports are supported safely; reject DT_HALF and all other
types explicitly, with exact shapes/counts checked before any memory copy.
This does NOT assert that an FP32 denoiser has been qualified.
Decoder FP32 input [1,128,3,23,40], output [1,3,17,720,1280] (already top
cropped by export). Copy through explicit plain CAFFE host tensors, validate
byte sizes, copy results and finite outputs. Request Precision_High. Check
actual execution backends using supported MNN APIs; fail on production CPU
fallback and report readiness honestly. Do not substitute another algorithm.
Avoid repeatedly loading all graphs in Java probe+load; nativeLoad should do
its own validation. Preserve safe handle lifetime and exception boundaries.
Add useful native error text if feasible without expanding this write scope.
Run relevant compiler/API checks. Do not commit or push; controller handles
the combined branch. Do not spawn subagents. Report exact files and limitations.

## Task 2: Java, pack and handset delivery

Controller aligns Java encode/noise with task 1, versions the executable pack
contract to reject obsolete prompt-based packs, updates pack builder and
meaningful tests. Fix misleading MobileI2V readiness/diagnostics and show
0.7.1, code 8. Add a concise handset acceptance note. Retain stable signing.

## Task 3: Build and verify

Review native and Java integration; run relevant Python and JVM tests and full
Android native/package build on GitHub Actions using a new branch. Verify
version, package, stable certificate, payload and SHA-256, download actual APK,
and deliver it with concise limitations and a single integrated handset check.

## Rulings

- The user has authorized repository work and APK delivery; no repeated
  permission request is needed for the isolated branch and CI build.
- Host qualification is not Adreno qualification; no fabricated GPU-ready pack.
- Existing source is a verified local snapshot of remote 1a8ef4b; remote writes
  use that commit as parent. Local snapshot commit is for review only.

- Integration correction: the measured encoder output is named `guide` (not the
  obsolete Android `latent` name). Device CAFFE_C4 is allowed with checked padded
  storage; only host transfer tensors must be plain CAFFE.
