package com.qujindai.localvideo;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class DepthMotionSpecTest {
    @Test
    public void nearLayersMoveMoreThanFarLayers() {
        float far = Math.abs(DepthMotionSpec.displayShiftX(
                DepthMotionSpec.Preset.PARALLAX_LEFT, 0.1f, 1000));
        float near = Math.abs(DepthMotionSpec.displayShiftX(
                DepthMotionSpec.Preset.PARALLAX_LEFT, 0.9f, 1000));
        assertTrue(near > far);
        assertTrue(near > 20f);
    }

    @Test
    public void leftAndRightHaveOppositeDirections() {
        float left = DepthMotionSpec.displayShiftX(
                DepthMotionSpec.Preset.PARALLAX_LEFT, 0.8f, 1000);
        float right = DepthMotionSpec.displayShiftX(
                DepthMotionSpec.Preset.PARALLAX_RIGHT, 0.8f, 1000);
        assertTrue(left < 0f);
        assertTrue(right > 0f);
    }

    @Test
    public void dollyZoomsNearLayersMore() {
        float far = DepthMotionSpec.zoomScale(DepthMotionSpec.Preset.DOLLY_IN, 0.1f);
        float near = DepthMotionSpec.zoomScale(DepthMotionSpec.Preset.DOLLY_IN, 0.9f);
        assertTrue(far > 1f);
        assertTrue(near > far);
    }
}
