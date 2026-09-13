# Third-party notices — Face-RE V0.9

The V0.9 APK is a single-purpose local static face-replacement application. Legacy RIFE, Depth Anything and MobileI2V components are not included in this APK build.

## Android Face Fusion reference

Face-RE adapts portions of the Android ONNX face-detection, alignment, ArcFace preprocessing and INSwapper preprocessing/post-processing approach from:

- Project: `Parasaran-Python/android-face-fusion`
- Pinned reference commit: `d5672622e53a97b04a158440a507126f384964ac`
- License: MIT License
- Copyright (c) 2026 Parasaran Vedanarayanan

The MIT license permits use, modification and distribution subject to preservation of the copyright and permission notice. Face-RE uses its own model import, fail-closed behavior, UI, result publishing and application structure.

## ONNX Runtime Android

Face-RE packages `com.microsoft.onnxruntime:onnxruntime-android:1.29.0`. ONNX Runtime is distributed under the MIT License.

## Face model weights

No SCRFD, ArcFace or INSwapper production model weights are committed to this repository or bundled in the APK. The user imports compatible model files separately. Model-weight usage and redistribution rights are separate from this application's source-code licenses; users are responsible for using model weights under terms that permit their intended use.
