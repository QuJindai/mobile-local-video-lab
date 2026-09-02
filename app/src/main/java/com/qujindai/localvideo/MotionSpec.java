package com.qujindai.localvideo;

public final class MotionSpec {
    public enum Preset {
        CINEMATIC_AUTO,
        PUSH_IN,
        PAN_LEFT,
        DRIFT_UP
    }

    public final float scale;
    public final float translateXRatio;
    public final float translateYRatio;
    public final float rotationDegrees;

    private MotionSpec(float scale, float translateXRatio,
                       float translateYRatio, float rotationDegrees) {
        this.scale = scale;
        this.translateXRatio = translateXRatio;
        this.translateYRatio = translateYRatio;
        this.rotationDegrees = rotationDegrees;
    }

    public static MotionSpec forPreset(Preset preset) {
        if (preset == null) preset = Preset.CINEMATIC_AUTO;
        switch (preset) {
            case PUSH_IN:
                return new MotionSpec(1.075f, 0f, -0.004f, 0f);
            case PAN_LEFT:
                return new MotionSpec(1.055f, -0.032f, -0.004f, 0.15f);
            case DRIFT_UP:
                return new MotionSpec(1.05f, 0.006f, -0.030f, -0.18f);
            case CINEMATIC_AUTO:
            default:
                return new MotionSpec(1.060f, -0.018f, -0.014f, 0.20f);
        }
    }
}
