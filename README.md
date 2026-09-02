# Mobile Local Video Lab

Android/Snapdragon local-video experiments for fully offline neural video generation.

## Current milestone

The first production milestone packages the open-source RIFE ncnn Vulkan inference core into an Android APK. It supports two local workflows:

- **Single image → motion clip**: the app synthesizes a small camera-motion endpoint locally and uses RIFE to generate neural intermediate frames.
- **Two images → interpolation clip**: the app uses RIFE directly to generate intermediate frames between two user-selected images.

All inference and MP4 encoding are designed to run on-device with no cloud API.

The RIFE/ncnn stage is the installable baseline for later MobileI2V/QNN work; it is intentionally labeled neural interpolation/motion generation rather than diffusion text-to-video.
