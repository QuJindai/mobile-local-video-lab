package com.qujindai.localvideo;

public final class DepthMotionSpec {
    public enum Preset {
        PARALLAX_LEFT,
        PARALLAX_RIGHT,
        DOLLY_IN,
        ARC_ORBIT,
        CRANE_UP,
        DOLLY_OUT
    }

    public enum Strength {
        GENTLE(0.55f, "轻柔"), NORMAL(1f, "标准"), STRONG(1.45f, "明显");
        public final float multiplier;
        public final String label;
        Strength(float multiplier, String label) {
            this.multiplier = multiplier;
            this.label = label;
        }
    }

    /** A camera pose shared by all depth layers, sampled once per keyframe. */
    public static final class Pose {
        private final float x, y, zoom, crop;
        private Pose(float x, float y, float zoom, float crop) {
            this.x = x;
            this.y = y;
            this.zoom = zoom;
            this.crop = crop;
        }
        public float shiftX(float depth, int width) {
            return x * (width - 1) * layer(depth);
        }
        public float shiftY(float depth, int height) {
            return y * (height - 1) * layer(depth);
        }
        public float zoomScale(float depth) {
            return crop + zoom * layer(depth);
        }
        private static float layer(float depth) {
            return .28f + .72f * clamp01(depth);
        }
    }

    public static Pose at(Preset preset, Strength strength, float time) {
        if (preset == null || strength == null || !Float.isFinite(time)) {
            throw new IllegalArgumentException("finite time, preset and strength required");
        }
        float t = clamp01(time);
        // Ease at both ends; all keyframes use the original image, never a previous warp.
        float u = t * t * (3f - 2f * t);
        float x = 0f, y = 0f, z = 0f, travel = 0f;
        switch (preset) {
            case PARALLAX_LEFT:
            case PARALLAX_RIGHT:
                x = (preset == Preset.PARALLAX_LEFT ? -.048f : .048f) * u;
                y = -.006f * u;
                z = .018f * u;
                travel = .048f;
                break;
            case DOLLY_IN:
                z = .09f * u;
                break;
            case ARC_ORBIT:
                x = .043f * (float) Math.sin(Math.PI * u);
                y = -.026f * (1f - (float) Math.cos(Math.PI * u)) * .5f;
                z = .035f * (float) Math.sin(Math.PI * u);
                travel = .043f;
                break;
            case CRANE_UP:
                x = .012f * (float) Math.sin(Math.PI * u);
                y = -.048f * u;
                z = .025f * u;
                travel = .048f;
                break;
            case DOLLY_OUT:
                z = .11f * (1f - u);
                break;
        }
        float amount = strength.multiplier;
        // Fixed per trajectory (avoids crop pumping), sufficient for either axis at any depth.
        return new Pose(x * amount, y * amount, z * amount, 1.002f + 2f * travel * amount);
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
