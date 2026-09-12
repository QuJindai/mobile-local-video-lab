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

The same qualifier accepts `--device cuda` for a CUDA reference/export host;
ONNX Runtime comparison still uses CPU. It rejects unavailable CUDA instead
of silently falling back. CUDA reports are host diagnostics and do not qualify
Adreno. Intermediate negative reports and forward timings preserve progress
when a long reference run cannot complete.

The qualifier always reports `android_gpu_pack_ready=false`. Even a successful
denoiser-only export still leaves the following acceptance incomplete:

1. VAE encoder/decoder conversion with matching sampling and crop semantics.
2. MNN conversion and numerical checks against the qualified reference.
3. Android inputs, output types and shapes matching the real exported graphs.
4. Actual Adreno execution evidence, memory and thermal measurements.
5. One integrated image-to-video, playback, save/share and diagnostics run.

Only then can the APK and model pack be delivered as a MobileI2V-enabled build.
There is no airplane-mode acceptance gate. Keep the stable baseline available.

## Measured VAE qualification, 2026-09-12

Both baseline VAE stages passed their full-size PyTorch/ONNX Runtime checks in
[run 34683171712](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34683171712),
at source commit `eef5315ffd3b4747fcda438936bcf7aa4171f024`. The exact reports are
stored in [encoder evidence](evidence/mobilei2v-vae-encoder.json) and
[decoder evidence](evidence/mobilei2v-vae-decoder.json). These measurements use
synthetic diagnostic inputs; they are not video-quality or handset evidence.

| Stage | Compared values | Maximum absolute error | RMSE | Result |
| --- | ---: | ---: | ---: | --- |
| Encoder, explicit posterior sample | 117,760 | 0.000001431 | 0.000000229 | Pass |
| Decoder, 17 frames cropped to 720p | 47,001,600 | 0.000016332 | 0.000001384 | Pass |

Both use FP32 and elementwise `atol=0.0001`, `rtol=0.001`. Strict loading covered
all 190 checkpoint tensors without missing or unexpected keys. The encoder
also passed explicit NumPy/PyTorch posterior-sampling parity. The observed raw
decoder shape was `[1,3,17,736,1280]` before the required top crop.

The first real run exposed a Torch 2.4 export defect: the symbolic lowering of
`movedim(-1, 1)` writes negative `Transpose.perm` entries rejected by ONNX Runtime.
The exporter now canonicalizes those axes to equivalent positive indices and
records every rewritten permutation in the reports. No weights, arithmetic,
precision, crop, or sampling were changed. The numerical comparison above was
performed after this correction.

VAE ONNX qualification is complete for these fixtures. Denoiser qualification,
MNN conversion, Android interface changes and actual Adreno execution remain
separate gates. The previous APK has not thereby become MobileI2V-ready.
