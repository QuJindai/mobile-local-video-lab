package com.qujindai.localvideo;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class MotionSpecTest {
    @Test
    public void pushInScalesWithoutLargePan() {
        MotionSpec spec = MotionSpec.forPreset(MotionSpec.Preset.PUSH_IN);
        assertTrue(spec.scale > 1.04f);
        assertTrue(Math.abs(spec.translateXRatio) < 0.01f);
        assertTrue(Math.abs(spec.translateYRatio) < 0.01f);
    }

    @Test
    public void panLeftMovesFrameLeftWithOverscan() {
        MotionSpec spec = MotionSpec.forPreset(MotionSpec.Preset.PAN_LEFT);
        assertTrue(spec.scale > 1.02f);
        assertTrue(spec.translateXRatio < -0.015f);
    }

    @Test
    public void driftUpMovesFrameUp() {
        MotionSpec spec = MotionSpec.forPreset(MotionSpec.Preset.DRIFT_UP);
        assertTrue(spec.translateYRatio < -0.015f);
    }

    @Test
    public void cinematicAutoCombinesZoomAndDiagonalDrift() {
        MotionSpec spec = MotionSpec.forPreset(MotionSpec.Preset.CINEMATIC_AUTO);
        assertTrue(spec.scale > 1.03f);
        assertTrue(Math.abs(spec.translateXRatio) > 0.005f);
        assertTrue(Math.abs(spec.translateYRatio) > 0.005f);
    }
}
