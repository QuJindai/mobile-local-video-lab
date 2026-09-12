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
Only a passing graph is uploaded as a qualified ONNX artifact. A failed graph
and its exact fixtures are now retained separately as `UNQUALIFIED-diagnostic`
for three days; they must not be packaged for Android.

`qualify_upstream.py` loads original model modules. On a CPU host it disables
only the optional CUDA/Triton import path; the selected LiteLA/GLUMBConv and
vanilla-attention implementations and learned weights remain the originals.
The report records this adaptation. This host diagnostic does not enable a CPU
production backend on Android.

The same qualifier accepts `--device cuda` for a CUDA reference/export host.
ONNX comparison defaults to CPU; `--onnx-provider cuda` requires
`onnxruntime-gpu==1.20.1`, checks actual CUDA initialization, and saves node
placement profiling with CPU/CUDA counts. A CUDA request fails when no CUDA
nodes execute. CUDA reports are host diagnostics and do not qualify Adreno.
Intermediate negative reports, exact input/reference fixtures, and forward
timings preserve progress when a long reference run cannot complete.

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

VAE ONNX qualification is complete for these fixtures. The MNN CPU result is
recorded below. Android interface changes and actual Adreno execution remain
separate gates. The previous APK has not thereby become MobileI2V-ready.

## Denoiser result and remaining blocker

[Run 34682248488](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34682248488)
at `8ecbf33dac2e7be7d52359d368913634d795fcfa` loaded all 267,218,048 learned
parameters and ran the real full-shape FP16 denoiser. Changing the unused prompt
had measured maximum difference `0.0`. ONNX export and graph execution succeeded,
but numerical parity **failed** with maximum absolute error `0.013671875` and
RMSE `0.0017619933933019638` under `atol=0.005`, `rtol=0.01`. The unmodified
[failure report](evidence/mobilei2v-denoiser-failed.json) is retained. No qualified
denoiser model was released and the tolerance has not been relaxed.

Subsequent review corrected the denoiser comparator to anchor relative tolerance
to the PyTorch reference, and made nonfinite prompt-perturbation failures retain
a valid negative JSON report. Ten lightweight validation tests now pass,
including a representable FP16 boundary case that was incorrectly accepted.
The historical report above is unchanged; no new full-model parity pass is
claimed from these helper tests.

The original ONNX ports are `latent`, `timestep`, `cond_mask`, `flow_score`, and
`output`; only `cond_mask` is FP32, with the others FP16. V0.7.1 removed Android's
obsolete `prompt`/`text_mask` inputs, added the encoder's explicit
`posterior_epsilon`, and consumes its `guide` output. All Android external ports
are FP32; serialized non-FP32 MNN inputs are rejected before initialization.
The original FP16 export therefore still cannot be imported as a runnable pack.

A source inspection identified a precision question requiring measurement:
[ORT 1.20.1's CPU cast transformer](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/optimizer/insert_cast_transformer.cc)
promotes unsupported FP16 operators to FP32 and eliminates inserted intermediate
casts. That changes rounding boundaries. This is a possible contributor to the
observed difference, not a proven attribution of this run's error. It does not
justify marking the failed comparison successful. A CUDA reference path is now
available in the qualifier; the attempted bounded Hugging Face T4 job did not
start because the service returned `402 Payment Required`.

Do not directly feed the current FP16 ONNX graph into the pinned MNN baseline:
its converter preserves `DT_HALF`, while `Tensor::setType` does not implement
that input type in this revision. VAE MNN qualification therefore accepts only
the verified FP32 VAE graphs and checks actual runtime port types before copying.
Denoiser precision, MNN conversion and real Adreno execution remain incomplete.
The [controlled precision experiment](MOBILEI2V_PRECISION_DIAGNOSIS.md) records
same-precision conversion error separately from changing FP16 arithmetic to
FP32, against the exact same half-rounded weights and inputs.

## Measured MNN VAE conversion, 2026-09-12

[Run 34684475670](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34684475670)
at `1ace2a33613a12f14d61718b6b2a9dbd63898695` converted both qualified FP32 VAE
graphs with the pinned MNN converter and ran both actual CPU forwards. Ports,
shapes, byte counts and finite outputs passed. Numerical parity **failed**:

| Stage | Maximum absolute error | RMSE | Result |
| --- | ---: | ---: | --- |
| Encoder | 0.000291176 | 0.000034164 | Fail |
| Decoder | 0.002969623 | 0.000084739 | Fail |

These use the same full-size PyTorch fixtures and unchanged elementwise
`atol=0.0001`, `rtol=0.001`. Original reports are retained in
[encoder evidence](evidence/mobilei2v-mnn-encoder-failed.json) and
[decoder evidence](evidence/mobilei2v-mnn-decoder-failed.json). The runtime used
MNN's AVX2 CPU extension (backend 13), `Precision_High`, four threads and the
default Winograd setting. Conversion and execution alone do not qualify these
models. The earlier backend classification defect has been corrected.

The follow-up [run 34685059248](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34685059248)
at `b7ef633da484ba231d61dc965e1df1baaacffce9` passed **both** VAE stages with
`WINOGRAD_MEMORY_LEVEL=0`. In the pinned CPU factory this selects dense
convolution instead of Winograd transforms. Source graphs, original weights,
full-size inputs and numerical tolerances remained unchanged.

| Stage with dense CPU convolution | Maximum absolute error | RMSE | Result |
| --- | ---: | ---: | --- |
| Encoder | 0.000002146 | 0.000000361 | Pass |
| Decoder | 0.000020236 | 0.000001500 | Pass |

Both reports verified artifact identities again after execution. The unmodified
[encoder report](evidence/mobilei2v-mnn-encoder-dense.json) and
[decoder report](evidence/mobilei2v-mnn-decoder-dense.json) record exact shapes,
hashes, tolerances, actual CPU backend and the disabled Winograd setting. The
workflow retains the successful converted models and exact fixtures as
`mobilei2v-mnn-encoder-host-only` and `mobilei2v-mnn-decoder-host-only` for seven
days. Failed models and outputs are retained separately for three days.

This qualifies the two VAE stages on the tested CPU fixtures with that setting.
It does not establish Adreno correctness, mobile speed/memory suitability or
generated-video quality. The next model blocker is denoiser precision and MNN
conversion. Its GPU diagnostic is prepared but did not start after the service
returned HTTP 402. The V0.7.1 handset test branch corrects Android contracts and delivers an APK
for existing-backend regression and diagnostics. It is not a MobileI2V-ready
release. Denoiser qualification, a complete compatible model pack, actual
Adreno execution and integrated image-to-video acceptance remain outstanding.
