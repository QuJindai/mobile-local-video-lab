package com.qujindai.localvideo;

public final class DepthMotionSpec {
    public enum Preset {
        PARALLAX_LEFT,
        PARALLAX_RIGHT,
        DOLLY_IN
    }

    private DepthMotionSpec() {}

    public static float displayShiftX(Preset preset, float depth, int width) {
        float d = clamp01(depth);
        float magnitude = width * (0.010f + 0.038f * d);
        if (preset == Preset.PARALLAX_LEFT) return -magnitude;
        if (preset == Preset.PARALLAX_RIGHT) return magnitude;
        return 0f;
    }

    public static float displayShiftY(Preset preset, float depth, int height) {
        float d = clamp01(depth);
        if (preset == Preset.DOLLY_IN) return -height * (0.002f + 0.004f * d);
        return -height * (0.002f + 0.006f * d);
    }

    public static float zoomScale(Preset preset, float depth) {
        float d = clamp01(depth);
        if (preset == Preset.DOLLY_IN) {
            return 1.018f + 0.072f * d;
        }
        return 1.018f + 0.018f * d;
    }

    private static float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }
}
