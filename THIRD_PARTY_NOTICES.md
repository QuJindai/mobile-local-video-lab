# Third-party notices — Face-RE V0.9.1

The V0.9.1 APK is a single-purpose static face-replacement application. Legacy RIFE, Depth Anything and MobileI2V components are not included in this APK build.

## Android Face Fusion reference

Face-RE adapts portions of the Android ONNX face-detection, alignment, ArcFace preprocessing and INSwapper preprocessing/post-processing approach from:

- Project: `Parasaran-Python/android-face-fusion`
- Pinned reference commit: `d5672622e53a97b04a158440a507126f384964ac`
- License: MIT License
- Copyright (c) 2026 Parasaran Vedanarayanan

The MIT license permits use, modification and distribution subject to preservation of the copyright and permission notice. Face-RE uses its own model download/import, fail-closed behavior, UI, result publishing and application structure.

## ONNX Runtime Android

Face-RE packages `com.microsoft.onnxruntime:onnxruntime-android:1.29.0`. ONNX Runtime is distributed under the MIT License.

## InsightFace pretrained model weights

Face-RE does **not** bundle SCRFD, ArcFace or INSwapper production weights in the APK. V0.9.1 can download them only after an explicit user action. The downloader supports the InsightFace GitHub model-zoo release and a China-friendly `hf-mirror.com` transport path; mirror downloads are accepted only when their content matches the pinned SHA-256 values.

Pinned model identities used by V0.9.1:

- `buffalo_l.zip`: `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`
- `det_10g.onnx`: `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`
- `w600k_r50.onnx`: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`
- `inswapper_128.onnx`: `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af`

`emap.bin` is generated locally from the downloaded INSwapper ONNX initializer; it is not fetched from a third party.

InsightFace states that its source code is MIT-licensed while its training data and pretrained models are subject to separate model-license terms and are available by default for non-commercial research purposes. Its current README directs users to contact InsightFace for commercial licensing of INSwapper-series and buffalo_l recognition models. The in-app first-download dialog surfaces this distinction before model retrieval.

Model downloading is the only Face-RE feature that requires Internet access. Source images, target images, face embeddings and swap results are processed locally and are not uploaded by Face-RE.
