# MobileI2V integration: model execution before APK delivery

The V0.7 APK is a runtime candidate. It has not completed MobileI2V generation
acceptance. RIFE/Depth3D clips cannot establish MobileI2V quality. The next
delivery depends on real checkpoint execution, conversion parity, then actual
accelerated handset inference.

## Reference artifacts

Use `tools/mobilei2v/upstream.lock.properties` for the denoiser and
`tools/mobilei2v/auxiliary.lock.properties` for the baseline VAE. The VAE
repository/subfolder is [author-confirmed](https://github.com/hustvl/MobileI2V/issues/2#issuecomment-3588124983).
Its immutable revision is selected for reproducible qualification, not claimed
as an author-issued training manifest. Verify downloaded bytes before loading.

The [author confirms](https://github.com/hustvl/MobileI2V/issues/11#issuecomment-4532698585)
that final MobileI2V omits text conditioning. The instantiated `SanaBlock_cross`
and `SanaBlock_vanila` forward methods in the pinned source do not use `y`.
Consequently, an exported graph must be inspected for its actual inputs; the
current Android requirement for `prompt` and `text_mask` is not authoritative.
This integration does not promise arbitrary text-directed actions.

## VAE rules that affect generated images

- Load the original `AutoencoderKLLTXVideo` using the pinned Diffusers 0.35.2
  implementation. Turbo-VAED is a separate, unqualified optimization.
- The loaded VAE configuration specifies scaling factor **1.0**. The unrelated
  `0.41407` YAML field must not replace the value actually used by the reference.
- Reference encoding samples the posterior. Export must preserve sampling with
  an explicit noise input or equivalent verified sampling outside the graph.
  A posterior-mean encoder is a different algorithm.
- Input image: `[1,3,1,720,1280]`; guide latent: `[1,128,1,23,40]`.
- Decoding `[1,128,3,23,40]` produces 17 frames with 736 rows. Match the upstream
  output by keeping rows **0:720**, not by resizing or center-cropping.
- The reference does not apply LTX per-channel latent mean/std normalization.

## Executable qualification

`MobileI2V Real Model Qualification` runs the original checkpoint in a CPU
diagnostic environment, with the exact source and weight hashes. It checks
learned-parameter loading, full-size finite outputs, measured prompt invariance,
the actual exported ONNX inputs, and ONNX Runtime/PyTorch numerical parity.
It uploads an ONNX artifact only after those checks pass.

`qualify_upstream.py` loads original model modules. On a CPU host it disables
only the optional CUDA/Triton import path; the selected LiteLA/GLUMBConv and
vanilla-attention implementations and learned weights remain the originals.
The report records this adaptation. This host diagnostic does not enable a CPU
production backend on Android.

The qualifier always reports `android_gpu_pack_ready=false`. Even a successful
denoiser-only export still leaves the following acceptance incomplete:

1. VAE encoder/decoder conversion with matching sampling and crop semantics.
2. MNN conversion and numerical checks against the qualified reference.
3. Android inputs, output types and shapes matching the real exported graphs.
4. Actual Adreno execution evidence, memory and thermal measurements.
5. One integrated image-to-video, playback, save/share and diagnostics run.

Only then can the APK and model pack be delivered as a MobileI2V-enabled build.
There is no airplane-mode acceptance gate. Keep the stable baseline available.
